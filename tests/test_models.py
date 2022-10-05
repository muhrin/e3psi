#!/usr/bin/env python3

from e3nn import o3
import torch

import e3psi
from e3psi import graphs
import mincepy
from mincepy import testing
from mincepy.testing import historian, mongodb_archive, archive_uri


def test_onsite_save_load(historian: mincepy.Historian):  # noqa: F811
    # Build the model
    site = e3psi.IrrepsObj(species=e3psi.Attr("4x0e"), pos=e3psi.Attr("1e"))

    torch.manual_seed(0)
    model = e3psi.OnsiteModel(site)

    # Create random inputs
    data = dict(site=site.irreps.randn(-1))
    ref_out = model(data)  # Reference output

    # Save/load a new model
    torch.manual_seed(0)
    loaded = testing.do_round_trip(historian, e3psi.OnsiteModel, site)

    # Check that all agrees
    assert loaded.graph == site
    assert torch.allclose(ref_out, loaded(data))


def test_intersite_save_load(historian: mincepy.Historian):  # noqa: F811
    # Build the model
    site1 = e3psi.IrrepsObj(species=e3psi.Attr("4x0e"), pos=e3psi.Attr("1e"))
    site2 = e3psi.IrrepsObj(species=e3psi.Attr("4x0e"), pos=e3psi.Attr("1e"))
    edge = e3psi.IrrepsObj(r=e3psi.Attr("0e"))
    graph = e3psi.TwoSite(site1, site2, edge)

    torch.manual_seed(0)
    model = e3psi.IntersiteModel(graph)

    # Create random inputs
    data = dict(
        site1=site1.irreps.randn(-1), site2=site2.irreps.randn(-1), edge=edge.irreps.randn(-1)
    )
    ref_out = model(data)  # Reference output

    # Save/load a new model
    torch.manual_seed(0)
    loaded = testing.do_round_trip(historian, e3psi.IntersiteModel, graph)

    # Check that all agrees
    assert loaded.graph == graph
    assert torch.allclose(ref_out, loaded(data))
