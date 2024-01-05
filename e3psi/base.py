import abc
import argparse
import functools
from typing import Union, Dict, Mapping, Any

from e3nn import o3
import torch

__all__ = "AbstractObj", "Attr", "IrrepsObj", "irreps", "create_tensor", "tensorial_attrs"


class AbstractObj:
    """An object that can create tensors whose values transform according to a set of irreps"""

    @property
    @abc.abstractmethod
    def irreps(self) -> o3.Irreps:
        """Get the attribute irreps"""

    @abc.abstractmethod
    def create_tensor(self, value, dtype=None, device=None) -> torch.Tensor:
        """Create an irrep tensor"""


class Attr(AbstractObj):
    def __init__(self, irreps) -> None:
        super().__init__()
        self._irreps = o3.Irreps(irreps)

    @property
    def irreps(self) -> o3.Irreps:
        return self._irreps

    def create_tensor(self, value, dtype=None, device=None) -> torch.Tensor:
        """Default implementation: creates torch tensor from the passed value"""
        return torch.tensor(value, dtype=dtype, device=device)

    def __eq__(self, other: Union[Any, "Attr"]) -> bool:
        if not isinstance(other, Attr):
            return False

        return self._irreps == other._irreps


class IrrepsObj(argparse.Namespace, AbstractObj):
    """An object that contains irrep attrs (can be e.g. a graph node, or edge)"""

    @property
    def irreps(self) -> o3.Irreps:
        ir = None
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            try:
                ir = val.irreps if ir is None else ir + val.irreps
            except AttributeError:
                raise AttributeError(f"Failed to get irreps for '{name}'")
        # return sum(list(val.irreps for val in vars(self.attrs).values()))
        return ir

    def create_tensor(self, values, dtype=None, device=None) -> torch.Tensor:
        tensors = tuple(
            create_tensor(attr, values[key], dtype=dtype, device=device)
            for key, attr in vars(self).items()
            if not key.startswith("_")
        )
        return torch.hstack(tensors)

    def __eq__(self, other: Union[Any, "IrrepsObj"]) -> bool:
        if not isinstance(other, IrrepsObj):
            return False

        return vars(self) == vars(other)


Tensorial = Union[Attr, IrrepsObj, o3.Irrep, o3.Irreps, str, dict]
ValueType = Union[torch.Tensor, Mapping[str, Any]]


@functools.singledispatch
def irreps(tensorial: Tensorial) -> o3.Irreps:
    """Get the irreps for a tensorial type"""
    raise TypeError(f"Unknown tensorial type: {tensorial.__class__.__name__}")


@irreps.register
def _(irreps_obj: IrrepsObj) -> o3.Irreps:
    total_irreps = None

    for name, val in tensorial_attrs(irreps_obj).items():
        try:
            val_irreps = irreps(val)
            # Concatenate
            total_irreps = val_irreps if total_irreps is None else total_irreps + val_irreps
        except AttributeError as exc:
            raise AttributeError(f"Failed to get irreps for {name}") from exc

    return total_irreps


@irreps.register
def _(attr: Attr) -> o3.Irreps:
    return attr.irreps


@irreps.register
def _(tensorial: o3.Irreps) -> o3.Irreps:
    return tensorial


@irreps.register
def _(tensorial: dict) -> o3.Irreps:
    """Irreps from a dictionary that only contains tensorial values"""
    return o3.Irreps("+".join([str(irreps(value)) for value in tensorial.values()]))


@irreps.register
def _(tensorial: o3.Irrep) -> o3.Irreps:
    """Convert an irrep into irreps"""
    return o3.Irreps([tensorial])


@functools.singledispatch
def create_tensor(tensorial, value: ValueType, dtype=None, device=None) -> torch.Tensor:
    """Create a tensor for a tensorial type"""
    raise TypeError(f"Unknown tensorial type: {tensorial.__class__.__name__}")


@create_tensor.register
def _(tensorial: IrrepsObj, value: ValueType, dtype=None, device=None) -> torch.Tensor:
    return torch.hstack(
        tuple(
            create_tensor(attr, value[key], dtype=dtype, device=device)
            for key, attr in tensorial_attrs(tensorial).items()
        )
    )


@create_tensor.register
def _(_tensorial: o3.Irreps, value: torch.Tensor, dtype=None, device=None) -> torch.Tensor:
    if not _tensorial.dim == torch.numel(value):
        raise ValueError(
            f"Irreps dimension ({_tensorial}) and value dimensions ({value.dim()}) do not match."
        )
    return value.to(dtype=dtype, device=device)


@create_tensor.register
def _(tensorial: dict, value: Mapping, dtype=None, device=None) -> torch.Tensor:
    return torch.hstack(
        tuple(
            create_tensor(attr, value[key], dtype=dtype, device=device)
            for key, attr in tensorial.items()
        )
    )


@create_tensor.register
def _(attr: Attr, value, dtype=None, device=None) -> torch.Tensor:
    return attr.create_tensor(value, dtype=dtype, device=device)


@create_tensor.register
def _(_tensorial: o3.Irrep, value, dtype=None, device=None) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=dtype, device=device)
    if not _tensorial.dim == torch.numel(value):
        raise ValueError(
            f"Irreps dimension ({_tensorial}) and value dimensions ({value.dim()}) do not match."
        )
    return value


def tensorial_attrs(irreps_obj: IrrepsObj) -> Dict[str, Tensorial]:
    """Get the irrep attributes for the passed object"""
    return {
        name: val
        for name, val in vars(irreps_obj).items()
        if not (name.startswith("_") or callable(val))
    }
