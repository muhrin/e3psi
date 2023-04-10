import abc
import argparse
from typing import Iterable
import uuid

from e3nn import o3
import mincepy
import torch

from . import base

__all__ = "SpecieOneHot", "OccuMtx", "OneSite", "TwoSite"


class OneSite:
    """Graph for a single (onsite) dataset"""

    def __init__(self, site: base.IrrepsObj):
        self.site = site

    def __eq__(self, other: "OneSite"):
        return self.site == other.site


class TwoSite:
    """A two site graph, useful for modelling two sites interacting with optional edge attributes"""

    edge = None

    def __init__(
        self, site1: base.IrrepsObj, site2: base.IrrepsObj, edge: base.IrrepsObj = None
    ) -> None:
        super().__init__()
        self.site1 = site1
        self.site2 = site2
        if edge:
            self.edge = edge

    def __eq__(self, other: "TwoSite"):
        return self.site1 == other.site1 and self.site2 == other.site2 and self.edge == other.edge


class SpecieOneHot(base.Attr):
    """Standard species one-hot encoding (direct sum of scalars)"""

    TYPE_ID = uuid.UUID("e4622421-e6cf-4ac3-89fe-9d967179e432")

    species = mincepy.field()

    def __init__(self, species: Iterable[str]) -> None:
        self.species = list(species)
        irreps = len(self.species) * o3.Irrep("0e")
        super().__init__(irreps)

    def create_tensor(self, specie, dtype=None, device=None) -> torch.Tensor:
        tens = torch.zeros(
            len(self.species), dtype=dtype or torch.get_default_dtype(), device=device
        )
        tens[self.species.index(specie)] = 1
        return tens


class OccuMtx(base.Attr):
    """Occupation matrix that will be represented as a direct sum of irreps"""

    TYPE_ID = uuid.UUID("50333915-35a4-48d0-ae52-531db72dee98")

    tp = mincepy.field()
    _tsq = None

    def __init__(self, orbital_irrep) -> None:
        self.tp = o3.ReducedTensorProducts("ij=ji", i=orbital_irrep)
        super().__init__(self.tp.irreps_out)

    def create_tensor(self, occ_mtx, dtype=None, device=None) -> torch.Tensor:
        occ = torch.tensor(occ_mtx, dtype=dtype or torch.get_default_dtype(), device=device)
        cob = self.tp.change_of_basis.to(dtype=dtype, device=device)
        return torch.einsum("zij,ij->z", cob, occ)

    def dist_euclidian(self, occs1, occs2) -> float:
        """Get the Euclidian distance between two occupation matrices"""
        return ((occs1 - occs2) ** 2).mean() ** 0.5

    def dist_cosine(self, occs1, occs2) -> float:
        """Get the cosine distance between the irrep vectors of the two occupation matrices"""
        occs1, occs2 = map(self.create_tensor, (occs1, occs2))
        return (
            1.0
            - (
                (occs1 * occs2).sum() / ((occs1 * occs1).sum() * (occs2 * occs2).sum()) ** 0.5
            ).item()
        )

    def dist_trace(self, occs1, occs2) -> float:
        """Get the distance that is the abs of the difference between the trace"""
        return abs(occs1.trace() - occs2.trace())

    def power_spectrum(self, occs):
        # Change basis to irrep basis
        if self._tsq is None:
            self._tsq = o3.TensorSquare(self.tp.irreps_out, filter_ir_out=["0e"])
        return self._tsq(self.create_tensor(occs))

    def dist_power_spectrum(self, occs1, occs2) -> float:
        # Calculate the power spectra
        occs1_ps = self.power_spectrum(occs1)
        occs2_ps = self.power_spectrum(occs2)
        # Take the RMSE between the two
        return torch.mean((occs1_ps - occs2_ps) ** 2).item() ** 0.5
