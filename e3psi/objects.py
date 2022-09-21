import abc
import argparse

from e3nn import o3
import torch

__all__ = "AbstractObj", "IrrepsObj", "Attr", "SpecieOneHot", "OccuMtx", "TwoSite"


class AbstractObj:
    """An object that can create tensors whose values transform according to a set of irreps"""

    @property
    @abc.abstractmethod
    def irreps(self) -> o3.Irreps:
        """Get the attribute irreps"""

    @abc.abstractmethod
    def create_tensor(self, value, dtype=None, device=None) -> torch.Tensor:
        """Create an irrep tensor"""


class IrrepsObj(AbstractObj, argparse.Namespace):
    """An object that contains irrep attrs (can be e.g. a graph node, edge, or whatever)"""

    @property
    def irreps(self) -> o3.Irreps:
        ir = None
        for val in vars(self).values():
            ir = val.irreps if ir is None else ir + val.irreps
        # return sum(list(val.irreps for val in vars(self.attrs).values()))
        return ir

    def create_tensor(self, values, dtype=None, device=None) -> torch.Tensor:
        return torch.hstack(
            tuple(
                attr.create_tensor(values[key], dtype=dtype, device=device)
                for key, attr in vars(self).items()
            )
        )


class TwoSite(IrrepsObj):
    """A two site graph, useful for modelling two sites interacting with optional edge attributes"""

    def __init__(self, site1: IrrepsObj, site2: IrrepsObj, edge: IrrepsObj = None) -> None:
        super().__init__()
        self.site1 = site1
        self.site2 = site2
        if edge:
            self.edge = edge


class Attr(AbstractObj):
    def __init__(self, irreps) -> None:
        self._irreps = o3.Irreps(irreps)

    @property
    def irreps(self) -> o3.Irreps:
        return self._irreps

    def create_tensor(self, value, dtype=None, device=None) -> torch.Tensor:
        """Generic version of creating a torch tensor"""
        return torch.tensor(value, dtype=dtype, device=device)


class SpecieOneHot(Attr):
    """Standard species one-hot encoding (direct sum of scalars)"""

    def __init__(self, species) -> None:
        self.species = list(species)
        irreps = len(self.species) * o3.Irrep("0e")
        super().__init__(irreps)

    def create_tensor(self, specie, dtype=None, device=None) -> torch.Tensor:
        tens = torch.zeros(len(self.species), dtype=dtype, device=device)
        tens[self.species.index(specie)] = 1
        return tens


class OccuMtx(Attr):
    """Occupation matrix that will be represented as a direct sum of irreps"""

    def __init__(self, orbital_irrep) -> None:
        self.tp = o3.ReducedTensorProducts("ij=ji", i=orbital_irrep)
        super().__init__(self.tp.irreps_out)

    def create_tensor(self, occ_mtx, dtype=None, device=None):
        occ = torch.tensor(occ_mtx, dtype=dtype, device=device)
        cob = self.tp.change_of_basis.to(dtype=dtype, device=device)
        return torch.einsum("zij,ij->z", cob, occ)
