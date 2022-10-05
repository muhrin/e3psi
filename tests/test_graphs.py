#!/usr/bin/env python3

from e3nn import o3
import torch

import e3psi
from e3psi import graphs
import mincepy
from mincepy import testing
from mincepy.testing import historian, mongodb_archive, archive_uri


def test_attr():
    irreps = o3.Irreps("4x0e")
    attr = e3psi.Attr(irreps)

    inp = irreps.randn(1, -1)
    assert torch.all(attr.create_tensor(inp) == inp)


def test_attr_save_load(historian: mincepy.Historian):  # noqa: F811
    irreps = o3.Irreps("4x0e")

    assert historian.hash(irreps) is not None
    loaded = testing.do_round_trip(historian, e3psi.Attr, irreps)
    assert loaded.irreps == irreps


def test_specie_one_hot():
    one_hot = e3psi.SpecieOneHot(["H", "N", "C", "F"])
    assert torch.all(one_hot.create_tensor("H") == torch.tensor([1, 0, 0, 0]))
    assert torch.all(one_hot.create_tensor("N") == torch.tensor([0, 1, 0, 0]))
    assert torch.all(one_hot.create_tensor("C") == torch.tensor([0, 0, 1, 0]))
    assert torch.all(one_hot.create_tensor("F") == torch.tensor([0, 0, 0, 1]))


def test_species_one_hot_save_load(historian: mincepy.Historian):  # noqa: F811
    species = ["H", "N", "C", "F"]
    loaded = testing.do_round_trip(historian, e3psi.SpecieOneHot, species)

    assert loaded.species == species


def test_irreps_obj_save_load(historian: mincepy.Historian):  # noqa: F811
    kwargs = {"species": e3psi.Attr("4x0e"), "pos": e3psi.Attr("1e")}
    loaded = testing.do_round_trip(historian, e3psi.IrrepsObj, **kwargs)

    assert loaded.species == kwargs["species"]
    assert loaded.pos == kwargs["pos"]


def test_occu_mtx_save_load(historian: mincepy.Historian):  # noqa: F811
    irrep = o3.Irreps("1e")
    input = torch.rand((3, 1))
    loaded = testing.do_round_trip(historian, e3psi.OccuMtx, irrep)

    occu_mtx = e3psi.OccuMtx(irrep)
    tensor = occu_mtx.create_tensor(input)
    assert torch.allclose(tensor, loaded.create_tensor(input))


def test_two_site_save_load(historian: mincepy.Historian):  # noqa: F811
    site1 = e3psi.IrrepsObj(species=e3psi.Attr("4x0e"), pos=e3psi.Attr("1e"))
    site2 = e3psi.IrrepsObj(species=e3psi.Attr("4x0e"), pos=e3psi.Attr("1e"))

    loaded = testing.do_round_trip(historian, e3psi.TwoSite, site1, site2)

    assert loaded.site1 == site1
    assert loaded.site2 == site2
