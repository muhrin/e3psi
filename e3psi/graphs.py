import abc
import argparse
from typing import Iterable
import uuid

from e3nn import o3
import mincepy
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


class IrrepsObj(argparse.Namespace, AbstractObj):
    """An object that contains irrep attrs (can be e.g. a graph node, or edge)"""

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


class IrrepsObjHelper(mincepy.BaseHelper):
    TYPE = IrrepsObj
    TYPE_ID = uuid.UUID("f8cd9a74-07d4-4a5e-9ed0-1bddfdcb94a4")

    def yield_hashables(self, obj: IrrepsObj, hasher):
        yield from hasher.yield_hashables(obj.__dict__)

    def save_instance_state(self, obj: IrrepsObj, _saver):
        return obj.__dict__

    def load_instance_state(self, obj: IrrepsObj, saved_state, _loader):
        for key, value in saved_state.items():
            setattr(obj, key, value)


class TwoSite(IrrepsObj):
    """A two site graph, useful for modelling two sites interacting with optional edge attributes"""

    TYPE_ID = uuid.UUID("576b54de-6708-415e-9860-886a4f93adac")

    def __init__(self, site1: IrrepsObj, site2: IrrepsObj, edge: IrrepsObj = None) -> None:
        super().__init__()
        self.site1 = site1
        self.site2 = site2
        if edge:
            self.edge = edge


class TwoSiteHelper(IrrepsObjHelper):
    TYPE = TwoSite
    TYPE_ID = uuid.UUID("29f0bcb3-a3dc-43f1-b739-50bec72d4ccc")


class Attr(mincepy.BaseSavableObject, AbstractObj):
    TYPE_ID = uuid.UUID("8a1832b6-0d11-4fe3-a7c2-5efada06b640")

    def __init__(self, irreps) -> None:
        super().__init__()
        self._irreps = o3.Irreps(irreps)

    @mincepy.field(attr="_irreps")
    def irreps(self) -> o3.Irreps:
        return self._irreps

    def create_tensor(self, value, dtype=None, device=None) -> torch.Tensor:
        return torch.tensor(value, dtype=dtype, device=device)


class SpecieOneHot(Attr):
    """Standard species one-hot encoding (direct sum of scalars)"""

    TYPE_ID = uuid.UUID("e4622421-e6cf-4ac3-89fe-9d967179e432")

    species = mincepy.field()

    def __init__(self, species: Iterable[str]) -> None:
        self.species = list(species)
        irreps = len(self.species) * o3.Irrep("0e")
        super().__init__(irreps)

    def create_tensor(self, specie, dtype=None, device=None) -> torch.Tensor:
        tens = torch.zeros(len(self.species), dtype=dtype, device=device)
        tens[self.species.index(specie)] = 1
        return tens


class OccuMtx(Attr):
    """Occupation matrix that will be represented as a direct sum of irreps"""

    TYPE_ID = uuid.UUID("50333915-35a4-48d0-ae52-531db72dee98")

    tp = mincepy.field()

    def __init__(self, orbital_irrep) -> None:
        self.tp = o3.ReducedTensorProducts("ij=ji", i=orbital_irrep)
        super().__init__(self.tp.irreps_out)

    def create_tensor(self, occ_mtx, dtype=None, device=None):
        occ = torch.tensor(occ_mtx, dtype=dtype, device=device)
        cob = self.tp.change_of_basis.to(dtype=dtype, device=device)
        return torch.einsum("zij,ij->z", cob, occ)


HISTORIAN_TYPES = (IrrepsObjHelper, TwoSiteHelper)
