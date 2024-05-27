import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars, here hardcoded in the init cuz we already know the output form of OneHot
        # Example: Irreps([(100, (0, 1)), (50, (1, 1))]) -> 100x0e+50x1e
        # AtomicDataDict.NODE_ATTRS_KEY is just a string in at global lvl, used to identify keys over the whole project
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])} # irreps out thus defines a vect (num_types x 0e)
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        # from GraphModuleMixin: all classes that extend GraphModuleMixin must call ``_init_irreps``
        # in their ``__init__`` functions with information on the data fields they expect, require, and produce, as well as their corresponding irreps.
        # setter for irreps_in/irreps_out
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # type(data): dict_keys(['edge_index', 'pos', 'batch', 'ptr', 'edge_cell_shift', 'cell', 'pbc', 'r_max', 'atom_types'])
        # AtomicDataDict.ATOM_TYPE_KEY = str('atom_types')

        # data encodes a pyg obj in dict shape.
        # it contains a batch of graphs ('pos' ownership indexed by 'batch')

        # feeds atom_types trhu one_hot encoding and modifies data in place by adding:
        # 'node_attrs' (and optionally 'node_features' aswell) as the retrieved one-hot encs

        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1) # single tensor with list of atom types over whole batch
        one_hot = torch.nn.functional.one_hot( # N_atoms x self.num_types
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data
