import numpy as np
import pytest
import tempfile
import torch
from os.path import isdir, isfile

from ase.data import chemical_symbols
from ase.io import write

from nequip.data import (
    AtomicData,
    AtomicDataDict,
    Collater,
    AtomicInMemoryDataset,
    NpzDataset,
    ASEDataset,
    HDF5Dataset,
    dataset_from_config,
    register_fields,
    deregister_fields,
)
from nequip.data.transforms import TypeMapper
from nequip.utils import Config


@pytest.fixture(scope="module")
def ase_file(molecules):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            write(fp.name, atoms, format="extxyz", append=True)
        yield fp.name


MAX_ATOMIC_NUMBER: int = 5
NATOMS = 10


@pytest.fixture(scope="function")
def npz():
    np.random.seed(0)
    natoms = NATOMS
    nframes = 8
    yield dict(
        positions=np.random.random((nframes, natoms, 3)),
        force=np.random.random((nframes, natoms, 3)),
        energy=np.random.random(nframes) * -600,
        Z=np.random.randint(1, MAX_ATOMIC_NUMBER, size=(nframes, natoms)),
    )


@pytest.fixture(scope="function")
def npz_data(npz):
    with tempfile.NamedTemporaryFile(suffix=".npz") as path:
        np.savez(path.name, **npz)
        yield path.name


@pytest.fixture(scope="function")
def npz_dataset(npz_data, temp_data):
    a = NpzDataset(
        file_name=npz_data,
        root=temp_data + "/test_dataset",
        AtomicData_options={"r_max": 3},
    )
    yield a


@pytest.fixture(scope="function")
def hdf5_dataset(npz, temp_data):
    try:
        import h5py
    except ModuleNotFoundError:
        pytest.skip("h5py is not installed")

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as path:
        f = h5py.File(path.name, "w")
        group = f.create_group("samples")
        group.create_dataset("atomic_numbers", data=npz["Z"], dtype=np.int8)
        group.create_dataset("pos", data=npz["positions"], dtype=np.float32)
        group.create_dataset("energy", data=npz["energy"], dtype=np.float32)
        group.create_dataset("forces", data=npz["force"], dtype=np.float32)
        yield HDF5Dataset(
            file_name=path.name,
            root=temp_data + "/test_dataset",
            AtomicData_options={"r_max": 3},
        )


@pytest.fixture(scope="function")
def root():
    with tempfile.TemporaryDirectory(prefix="datasetroot") as path:
        yield path


def test_type_mapper():
    tm = TypeMapper(chemical_symbol_to_type={"C": 1, "H": 0})
    atomic_numbers = torch.as_tensor([1, 1, 6, 1, 6, 6, 6])
    atom_types = tm.transform(atomic_numbers)
    assert atom_types[0] == 0
    untransformed = tm.untransform(atom_types)
    assert torch.equal(untransformed, atomic_numbers)


class TestInit:
    def test_init(self):
        with pytest.raises(NotImplementedError) as excinfo:
            AtomicInMemoryDataset(root=None)
        assert str(excinfo.value) == ""

    def test_npz(self, npz_data, root):
        g = NpzDataset(file_name=npz_data, root=root, AtomicData_options={"r_max": 3.0})
        assert isdir(g.root)
        assert isdir(g.processed_dir)
        assert isfile(g.processed_dir + "/data.pth")

    def test_ase(self, ase_file, root):
        a = ASEDataset(
            file_name=ase_file,
            root=root,
            AtomicData_options={"r_max": 3.0},
            ase_args=dict(format="extxyz"),
        )
        assert isdir(a.root)
        assert isdir(a.processed_dir)
        assert isfile(a.processed_dir + "/data.pth")


class TestStatistics:
    @pytest.mark.xfail(
        reason="Current subset hack doesn't support statistics of non-per-node callable"
    )
    def test_callable(self, npz_dataset, npz):
        # Get componentwise statistics
        ((f_mean, f_std),) = npz_dataset.statistics(
            [lambda d: torch.flatten(d[AtomicDataDict.FORCE_KEY])]
        )
        n_ex, n_at, _ = npz["force"].shape
        f_raveled = npz["force"].reshape((n_ex * n_at * 3,))
        assert np.allclose(np.mean(f_raveled), f_mean)
        # By default we follow torch convention of defaulting to the unbiased std
        assert np.allclose(np.std(f_raveled, ddof=1), f_std)

    @pytest.mark.parametrize("dataset_type", ["npz_dataset", "hdf5_dataset"])
    def test_statistics(self, dataset_type, npz, request):
        dataset = request.getfixturevalue(dataset_type)
        (eng_mean, eng_std), (Z_unique, Z_count) = dataset.statistics(
            fields=[AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.ATOMIC_NUMBERS_KEY],
            modes=["mean_std", "count"],
        )

        eng = npz["energy"]
        assert np.allclose(eng_mean, np.mean(eng))
        # By default we follow torch convention of defaulting to the unbiased std
        assert np.allclose(eng_std, np.std(eng, ddof=1))

        if isinstance(Z_count, torch.Tensor):
            Z_count = Z_count.numpy()
            Z_unique = Z_unique.numpy()

        uniq, count = np.unique(npz["Z"].ravel(), return_counts=True)
        assert np.all(Z_unique == uniq)
        assert np.all(Z_count == count)

    @pytest.mark.parametrize("dataset_type", ["npz_dataset", "hdf5_dataset"])
    def test_with_subset(self, dataset_type, npz, request):
        dataset = request.getfixturevalue(dataset_type).index_select([0])

        ((Z_unique, Z_count), (force_rms,)) = dataset.statistics(
            [AtomicDataDict.ATOMIC_NUMBERS_KEY, AtomicDataDict.FORCE_KEY],
            modes=["count", "rms"],
        )

        uniq, count = np.unique(npz["Z"][0].ravel(), return_counts=True)
        assert np.all(Z_unique.numpy() == uniq)
        assert np.all(Z_count.numpy() == count)

        assert np.allclose(
            force_rms.numpy(), np.sqrt(np.mean(np.square(npz["force"][0])))
        )

    @pytest.mark.parametrize("dataset_type", ["npz_dataset", "hdf5_dataset"])
    def test_atom_types(self, dataset_type, request):
        dataset = request.getfixturevalue(dataset_type)
        ((avg_num_neigh, _),) = dataset.statistics(
            fields=[
                lambda data: (
                    torch.unique(
                        data[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True
                    )[1],
                    "node",
                )
            ],
            modes=["mean_std"],
        )
        # They are all homogenous in this dataset:
        assert (
            avg_num_neigh
            == torch.bincount(dataset[0][AtomicDataDict.EDGE_INDEX_KEY][0])[0]
        )

    @pytest.mark.parametrize("dataset_type", ["npz_dataset", "hdf5_dataset"])
    def test_edgewise_stats(self, dataset_type, request):
        dataset = request.getfixturevalue(dataset_type)
        ((avg_edge_length, std_edge_len),) = dataset.statistics(
            fields=[
                lambda data: (
                    (
                        data[AtomicDataDict.POSITIONS_KEY][
                            data[AtomicDataDict.EDGE_INDEX_KEY][1]
                        ]
                        - data[AtomicDataDict.POSITIONS_KEY][
                            data[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ]
                    ).norm(dim=-1),
                    "edge",
                )
            ],
            modes=["mean_std"],
        )
        collater = Collater.for_dataset(dataset)
        all_data = collater([dataset[i] for i in range(len(dataset))])
        all_data = AtomicData.to_AtomicDataDict(all_data)
        all_data = AtomicDataDict.with_edge_vectors(all_data, with_lengths=True)
        assert torch.allclose(
            avg_edge_length,
            torch.mean(all_data[AtomicDataDict.EDGE_LENGTH_KEY]).to(
                avg_edge_length.dtype
            ),
        )
        assert torch.allclose(
            std_edge_len,
            torch.std(all_data[AtomicDataDict.EDGE_LENGTH_KEY]).to(
                avg_edge_length.dtype
            ),
        )


class TestPerAtomStatistics:
    @pytest.mark.parametrize("mode", ["mean_std", "rms"])
    def test_per_node_field(self, npz_dataset, mode):
        # set up the transformer
        npz_dataset = set_up_transformer(npz_dataset, True, False)

        with pytest.raises(ValueError) as excinfo:
            npz_dataset.statistics(
                [AtomicDataDict.BATCH_KEY],
                modes=[f"per_atom_{mode}"],
            )
            assert (
                excinfo
                == f"It doesn't make sense to ask for `{mode}` since `{AtomicDataDict.BATCH_KEY}` is not per-graph"
            )

    @pytest.mark.parametrize("subset", [True, False])
    @pytest.mark.parametrize(
        "key,dim", [(AtomicDataDict.TOTAL_ENERGY_KEY, (1,)), ("somekey", (3,))]
    )
    def test_per_graph_field(self, npz_dataset, subset, key, dim):
        if key == "somekey":
            register_fields(graph_fields=[key])

        npz_dataset = set_up_transformer(npz_dataset, True, subset)
        if npz_dataset is None:
            return

        rng = torch.Generator().manual_seed(454)
        E = torch.rand((npz_dataset.len(),) + dim, generator=rng)
        ref_mean = torch.mean(E / NATOMS, dim=0)
        ref_std = torch.std(E / NATOMS, dim=0)

        if subset:
            E_orig_order = torch.zeros(
                (npz_dataset.data[AtomicDataDict.TOTAL_ENERGY_KEY].shape[0],) + dim
            )
            E_orig_order[npz_dataset._indices] = E
            npz_dataset.data[key] = E_orig_order
        else:
            npz_dataset.data[key] = E

        ((mean, std),) = npz_dataset.statistics(
            [key],
            modes=["per_atom_mean_std"],
        )

        print("mean", mean, ref_mean)
        print("diff in mean", mean - ref_mean)
        print("std", std, ref_std)

        assert torch.allclose(mean, ref_mean, rtol=1e-1)
        assert torch.allclose(std, ref_std, rtol=1e-2)

        if key == "somekey":
            deregister_fields(key)


class TestPerSpeciesStatistics:
    @pytest.mark.parametrize("mode", ["mean_std", "rms"])
    @pytest.mark.parametrize("subset", [True, False])
    def test_per_node_field(self, npz_dataset, mode, subset):
        # set up the transformer
        npz_dataset = set_up_transformer(npz_dataset, True, subset)

        (result,) = npz_dataset.statistics(
            [AtomicDataDict.BATCH_KEY],
            modes=[f"per_species_{mode}"],
        )
        print(result)

    @pytest.mark.parametrize("alpha", [0, 1e-3, 0.01])
    @pytest.mark.parametrize("full_rank", [True, False])
    @pytest.mark.parametrize("subset", [True, False])
    def test_per_graph_field(self, npz_dataset, alpha, full_rank, subset):
        if alpha <= 1e-4 and not full_rank:
            return

        npz_dataset = set_up_transformer(npz_dataset, full_rank, subset)
        if npz_dataset is None:
            return

        # get species count per graph
        Ns = []
        for i in range(npz_dataset.len()):
            Ns.append(
                torch.bincount(npz_dataset[i][AtomicDataDict.ATOM_TYPE_KEY].view(-1))
            )
        n_spec = max(len(e) for e in Ns)
        N = torch.zeros(len(Ns), n_spec)
        for i in range(len(Ns)):
            N[i, : len(Ns[i])] = Ns[i]
        del n_spec
        del Ns

        ref_mean, ref_std, E = generate_E(N, 100, 1000, 10)

        if subset:
            E_orig_order = torch.zeros_like(
                npz_dataset.data[AtomicDataDict.TOTAL_ENERGY_KEY]
            )
            E_orig_order[npz_dataset._indices] = E.unsqueeze(-1)
            npz_dataset.data[AtomicDataDict.TOTAL_ENERGY_KEY] = E_orig_order
        else:
            npz_dataset.data[AtomicDataDict.TOTAL_ENERGY_KEY] = E

        ref_res2 = torch.square(
            torch.matmul(N, ref_mean.reshape([-1, 1])) - E.reshape([-1, 1])
        ).sum()

        ((mean, std),) = npz_dataset.statistics(
            [AtomicDataDict.TOTAL_ENERGY_KEY],
            modes=["per_species_mean_std"],
            kwargs={
                AtomicDataDict.TOTAL_ENERGY_KEY
                + "per_species_mean_std": {
                    "alpha": alpha,
                    "stride": 1,
                }
            },
        )

        res = torch.matmul(N, mean.reshape([-1, 1])) - E.reshape([-1, 1])
        res2 = torch.sum(torch.square(res))
        print("alpha, residue, actual residue", alpha, res2, ref_res2)
        print("mean", mean, ref_mean)
        print("diff in mean", mean - ref_mean)
        print("std", std, ref_std)

        tolerance = torch.max(ref_std) * 4
        if full_rank:
            assert torch.allclose(mean, ref_mean, atol=tolerance)
            # assert torch.allclose(std, torch.zeros_like(ref_mean), atol=alpha * 100)
        else:
            assert torch.allclose(mean, mean[0], atol=tolerance)
            # assert torch.std(mean).numpy() == 0


class TestReload:
    @pytest.mark.parametrize("change_rmax", [0, 1])
    @pytest.mark.parametrize("give_url", [True, False])
    @pytest.mark.parametrize("change_key_map", [True, False])
    def test_reload(self, npz_dataset, npz_data, change_rmax, give_url, change_key_map):
        r_max = npz_dataset.AtomicData_options["r_max"] + change_rmax
        keymap = npz_dataset.key_mapping.copy()  # the default one
        if change_key_map:
            keymap["x1"] = "x2"
        a = NpzDataset(
            file_name=npz_data,
            root=npz_dataset.root,
            AtomicData_options={"r_max": r_max},
            key_mapping=keymap,
            **({"url": "example.com/data.dat"} if give_url else {}),
        )
        print(a.processed_file_names[0])
        print(npz_dataset.processed_file_names[0])
        assert (a.processed_dir == npz_dataset.processed_dir) == (
            (change_rmax == 0) and (not give_url) and (not change_key_map)
        )


class TestFromConfig:
    @pytest.mark.parametrize(
        "args",
        [
            dict(AtomicData_options={"r_max": 3.0}),
            dict(dataset_AtomicData_options={"r_max": 3.0}),
            dict(r_max=3.0),
            dict(r_max=3.0, AtomicData_options={}),
        ],
    )
    def test_npz(self, npz_data, root, args):
        config = Config(
            dict(
                dataset="npz",
                file_name=npz_data,
                root=root,
                chemical_symbol_to_type={
                    chemical_symbols[an]: an - 1 for an in range(1, MAX_ATOMIC_NUMBER)
                },
                **args,
            )
        )
        g = dataset_from_config(config)
        assert g.AtomicData_options["r_max"] == 3
        assert isdir(g.root)
        assert isdir(g.processed_dir)
        assert isfile(g.processed_dir + "/data.pth")

    @pytest.mark.parametrize("prefix", ["dataset", "thingy"])
    def test_ase(self, ase_file, root, prefix):
        config = Config(
            dict(
                file_name=ase_file,
                root=root,
                AtomicData_options={"r_max": 3.0},
                ase_args=dict(format="extxyz"),
                chemical_symbol_to_type={"H": 0, "C": 1, "O": 2},
            )
        )
        config[prefix] = "ASEDataset"
        a = dataset_from_config(config, prefix=prefix)
        assert isdir(a.root)
        assert isdir(a.processed_dir)
        assert isfile(a.processed_dir + "/data.pth")

        # Test reload
        # Change some random ASE specific parameter
        # See https://wiki.fysik.dtu.dk/ase/ase/io/io.html
        config["ase_args"]["do_not_split_by_at_sign"] = True
        b = dataset_from_config(config, prefix=prefix)
        assert isdir(b.processed_dir)
        assert isfile(b.processed_dir + "/data.pth")
        assert a.processed_dir != b.processed_dir


class TestFromList:
    def test_from_atoms(self, molecules):
        dataset = ASEDataset.from_atoms_list(
            molecules, AtomicData_options={"r_max": 4.5}
        )
        assert len(dataset) == len(molecules)
        for i, mol in enumerate(molecules):
            assert np.array_equal(
                mol.get_atomic_numbers(), dataset[i].to_ase().get_atomic_numbers()
            )


def generate_E(N, mean_min, mean_max, std):
    rng = torch.Generator().manual_seed(568)
    ref_mean = (
        torch.rand((N.shape[1]), generator=rng) * (mean_max - mean_min) + mean_min
    )
    t_mean = torch.ones((N.shape[0], 1)) * ref_mean.reshape([1, -1])
    ref_std = torch.rand((N.shape[1]), generator=rng) * std
    t_std = torch.ones((N.shape[0], 1)) * ref_std.reshape([1, -1])
    E = torch.normal(t_mean, t_std, generator=rng)
    return ref_mean, ref_std, (N * E).sum(axis=-1)


def set_up_transformer(npz_dataset, full_rank, subset):
    if full_rank:
        unique = torch.unique(npz_dataset.data[AtomicDataDict.ATOMIC_NUMBERS_KEY])
        npz_dataset.transform = TypeMapper(
            chemical_symbol_to_type={
                chemical_symbols[n]: i for i, n in enumerate(unique)
            }
        )
    else:
        ntype = 2

        # let all atoms to be the same type distribution
        num_nodes = npz_dataset.data[AtomicDataDict.BATCH_KEY].shape[0]
        new_n = torch.ones(num_nodes, dtype=torch.int64)
        new_n[::NATOMS] += ntype
        npz_dataset.data[AtomicDataDict.ATOMIC_NUMBERS_KEY] = new_n

        # set up the transformer
        npz_dataset.transform = TypeMapper(
            chemical_symbol_to_type={
                chemical_symbols[n]: i for i, n in enumerate([1, ntype + 1])
            }
        )
    if subset:
        return npz_dataset.index_select(torch.randperm(len(npz_dataset)))
    else:
        return npz_dataset
