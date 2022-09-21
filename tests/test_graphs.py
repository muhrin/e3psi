#!/usr/bin/env python3

from e3nn import o3
import torch

import e3psi


def test_attr():
    irreps = o3.Irreps("4x0e")
    attr = e3psi.Attr(irreps)

    inp = irreps.randn(1, -1)
    assert torch.all(attr.create_tensor(inp) == inp)


def test_specie_one_hot():
    one_hot = e3psi.SpecieOneHot(["H", "N", "C", "F"])
    assert torch.all(one_hot.create_tensor("H") == torch.tensor([1, 0, 0, 0]))
    assert torch.all(one_hot.create_tensor("N") == torch.tensor([0, 1, 0, 0]))
    assert torch.all(one_hot.create_tensor("C") == torch.tensor([0, 0, 1, 0]))
    assert torch.all(one_hot.create_tensor("F") == torch.tensor([0, 0, 0, 1]))
