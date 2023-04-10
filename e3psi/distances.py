from e3nn import o3
import torch

from . import base


class PowerSpectrumDistance:
    _tsq = None  # This will be lazily created when needed

    def __init__(self, tensorial: base.Tensorial) -> None:
        self._tensorial = tensorial
        self.irreps_in = base.irreps(tensorial)

    def power_spectrum(self, value: base.ValueType):
        """Calculate the power spectrum for the passed tensor"""
        if self._tsq is None:
            self._tsq = o3.TensorSquare(self.irreps_in, filter_ir_out=["0e"])

        tensor = base.create_tensor(self._tensorial, value)
        self._tsq.to(device=tensor.device, dtype=tensor.dtype)
        return self._tsq(tensor)

    def get_distance(self, tensor1: base.ValueType, tensor2: base.ValueType) -> float:
        # Calculate the power spectra
        ps1 = self.power_spectrum(tensor1)
        ps2 = self.power_spectrum(tensor2)
        return self.get_distance_from_ps(ps1, ps2)

    def get_distance_from_ps(self, ps1, ps2) -> float:
        # Take the RMSD between the two
        return torch.mean((ps1 - ps2) ** 2).item() ** 0.5
