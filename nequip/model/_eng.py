from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from . import builder_utils


def SimpleIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps.
    First builder called when the default model gets instanciated

    The goal of this funct is to load simple_irreps_keys, real_irreps_keys.

    if real_irreps_keys are provided then return, no preprocessing is done
    if simple_irreps_keys are provided then generate shapes and check pre-condition
    """

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden

    # From line 41 of full.yaml:
    # simple_irreps_keys: automatic construction of net shapes
    # real_irreps_keys: custom net shapes
    # the irreps of the features in various parts of the network can be specified directly via real_irreps_keys
    # use either simple_irreps_keys, or real_irreps_keys, one of the two should be provided--- they cannot be mixed.

    # looks for whether simple_irreps_keys have been provided real_irreps_keys
    prefix = "" if prefix is None else f"{prefix}_"

    real_irreps_keys   = [ # line 4 of full.yaml
        "chemical_embedding_irreps_out",   # irreps for the chemical embedding of species
        "feature_irreps_hidden",           # hidden layer irreps
        "irreps_edge_sh",                  # SH irreps to embed edges. If a single integer, indicates the full SH up to L_max=that_integer  #
        "conv_to_output_hidden_irreps_out",
        #? irreps used in hidden layer of output block  # irreps in out requested, in nequip it is potential energy -> even scalar 1x0e
        # nb vects have odd parity
    ]

    has_full: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys
    )

    if has_full: # nothing to do if not has_simple
        return

    simple_irreps_keys = [ # line 37 of full.yaml
        "l_max",
        "parity",
        "num_features"
    ]

    has_simple: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys
    )

    assert has_simple

    # automatize hidden irreps shape generation
    update = {}
    lmax = config.get(f"{prefix}l_max", config["l_max"])
    parity = config.get(f"{prefix}parity", config["parity"])
    num_features = config.get(f"{prefix}num_features", config["num_features"])

    update[f"{prefix}chemical_embedding_irreps_out"] = repr(
        o3.Irreps([(num_features, (0, 1))])  # n scalars
    )

    update[f"{prefix}irreps_edge_sh"] = repr(
        o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
    )

    update[f"{prefix}feature_irreps_hidden"] = repr(
        o3.Irreps(
            [
                (num_features, (l, p))
                for p in ((1, -1) if parity else (1,))
                for l in range(lmax + 1)
            ]
        )
    )

    update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
        # num_features // 2  scalars
        o3.Irreps([(max(1, num_features // 2), (0, 1))])
    )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert (
                config[k] == v
            ), f"For key {k}, the full irreps options had value `{config[k]}` inconsistant with the value derived from the simple irreps options `{v}`"
        config[k] = v


def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    Main tutorial: https://deepnote.com/app/shuai-jiang-c648/NequIP-Tutorial-Duplicate-b4c9a903-0586-4587-a353-85181f017467

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    # config is modified in-place
    builder_utils.add_avg_num_neighbors( # updates config obj using dataset
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3) # getter and default value if key not present

    # EMBEDDING LAYERS
    layers = {
        # -- Encode --
        "one_hot":      OneHotAtomEncoding,         # module that in its forward set node features to one hot
        "spharm_edges": SphericalHarmonicEdgeAttrs, # project edg displacement vect on SH; lmax=3; project angular part
        "radial_basis": RadialBasisEdgeEncoding,    # project edg displacement vect on BBF; embeds radial part of displacements
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,       # from 1-hot to 32 scalars here, it acts exactly like a traditional MLP
        # in its forward # data[self.field] is the oneHot of atom type, self.field = 'node_features'
        # AtomiwiseLinear for an initial chemical embedding of each atom:
        # applied independently but with *shared weights* on all atoms in the system,
        # it does not communicate information between atoms.
        # We use this first layer to take us from the space of species one-hots into the feature space, specified by chemical_embedding_irreps_out

        # AtomwiseLinear ->
        # idea: eg given a "2x0e + 2x1o" as input and a desired output of shape we have:
        # the "2x0e" part is processed by a traditional MLP cuz just scalars, while the "2x1o" part is processed in an equivariant manner:
        # given the 2 input vectors v1,v2 in the geom. tensor we have 4 weights w11, w12, w21, w22 s.t.:
        # out1 = w11*v1 + w12*v2 and out2 = w21*v1 + w22*v2
        # then activate via gating (scalar can be activated directly via relu), L=1 part is activated via gating
        # example from Linear docstring:
        # >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
        # >>> lin.weight_numel
        # 160

    }

    # INTERACTION BLOCK fig 2 of paper, center
    # add convnet layers
    # insertion preserves order
    # ConvNetLayer:
    # a) an AtomwiseLinear, <-> self interaction
    # b) equivariant convolution,
    # c) another AtomwiseLinear, self interaction,
    # d) a nonlinearity (Gate)
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # OUTPUT BLOCK
    # the l = 0 features of the final ConvNetLayer are passed to the output block block, which consists of a set of
    # .update also maintains insertion order
    # After a series of convolutions, we want to go from the learned features to the atomic outputs.
    layers.update( # two atom-wise self-interaction layers.
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear, # an intermediate step to reduce the feature dimension
            "output_hidden_to_scalar": ( # This takes us from the feature representation to the single scalar output which we interpret as the atomic potential energy
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    # For each atom the final layer outputs a single scalar, which is the atomic potential energy.
    # These are then summed to give the total predicted potential energy of the sys
    # Forces are subsequently obtained as the negative gradient of the predicted total potential energy
    layers["total_energy_sum"] = (
        AtomwiseReduce, # sums over atomic outputs, this gives us the total predicted potential energy, a sum over atomic potential energies.
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    # SequentialGraphNetwork a class that build a network through a Sequential of layers, all of which input and output a graph.
    # instanciate SequentialGraphNetwork and returns it
    return SequentialGraphNetwork.from_parameters(
        shared_params=config, # model params
        layers=layers, # dict of layer_name:layer_obj
    )
