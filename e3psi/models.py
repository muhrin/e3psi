import abc
import uuid
from typing import Union, Tuple

import mincepy
from mincepy_sci import pytorch_types
import mincepy_sci.pytorch_types
from e3nn import nn
from e3nn import o3
import torch

from . import graphs

__all__ = "OnsiteModel", "IntersiteModel", "Model"


class Module(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class Compose(Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    @property
    def irreps_in(self):
        return self.first.irreps_in

    @property
    def irreps_out(self):
        return self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class Model(Module, pytorch_types.SavableModuleMixin, abc.ABC):
    @property
    @abc.abstractmethod
    def graph(self) -> graphs.AbstractObj:
        """Get the graph that described this model"""

    def __hash__(self):
        return object.__hash__(self)


class OnsiteModel(Model):
    """Model for predicting onsite values i.e. a node with self interactions"""

    def __init__(
        self,
        graph: graphs.AbstractObj,
        nn_irreps_out: Union[str, o3.Irreps] = None,
        irreps_out="0e",
        irrep_normalization="component",
        hidden_layers=1,
        rescaler=None,
    ):
        super().__init__()

        self._graph = graph
        # self.irreps_in = o3.Irrep('0e') + self._graph.irreps
        self.irreps_in = self._graph.irreps
        feature_irreps = (
            o3.Irreps(nn_irreps_out)
            if nn_irreps_out is not None
            else o3.ReducedTensorProducts("ij=ji", i=self.irreps_in).irreps_out
        )

        # First layer
        self.layers = torch.nn.ModuleList(
            _self_interaction(
                self.irreps_in, feature_irreps, irrep_normalization=irrep_normalization
            )
        )

        if hidden_layers > 1:
            # Intermediate hidden
            for _ in range(hidden_layers - 1):
                self.layers.extend(
                    _self_interaction(
                        self.layers[-1].irreps_out,
                        feature_irreps,
                        irrep_normalization=irrep_normalization,
                    )
                )

        # Output
        self.layers.append(
            o3.TensorSquare(
                self.layers[-1].irreps_out,
                irreps_out=irreps_out,
                irrep_normalization=irrep_normalization,
            )
        )

        if rescaler is not None:
            self.layers.append(rescaler)

    @property
    def graph(self):
        return self._graph

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs["site"]
        # if len(out.shape) == 2:
        #     one = torch.ones(out.shape[0], 1, dtype=out.dtype, device=out.device)
        # else:
        #     one = torch.ones(1, dtype=out.dtype, device=out.device)
        #
        # out = torch.hstack((one, out))

        for layer in self.layers:
            out = layer(out)

        return out


class IntersiteModel(Model):
    """Model for predicting intersite values, i.e. a two-body property"""

    TYPE_ID = uuid.UUID("8d3f024e-c9d7-48d0-92e1-88448d483936")

    def __init__(
        self,
        graph: graphs.TwoSite,
        n1n2_irreps_out=None,
        n1n2e_irreps_out=None,
        irreps_out="0e",
        hidden_layers=1,
        irrep_normalization="component",
        rescaler=None,
    ):
        super().__init__()
        self._graph = graph
        self.layers = torch.nn.ModuleList()

        # Node-node TP
        node_node_tp_irreps_out = (
            o3.Irreps(n1n2_irreps_out)
            if n1n2_irreps_out is not None
            else o3.FullTensorProduct(self.graph.site1.irreps, self.graph.site2.irreps).irreps_out
        )

        # Input layers
        self.node_node_tp = o3.FullyConnectedTensorProduct(
            self.graph.site1.irreps,
            self.graph.site2.irreps,
            irreps_out=node_node_tp_irreps_out,
            irrep_normalization="component",
        )

        # Node-node output * edge TP
        node_node_edge_tp_irreps_out = (
            o3.Irreps(n1n2e_irreps_out)
            if n1n2e_irreps_out is not None
            else o3.FullTensorProduct(
                self.node_node_tp.irreps_out, self.graph.edge.irreps
            ).irreps_out
        )

        self.node_node_edge_tp, gate = _interaction(
            node_node_tp_irreps_out,
            self.graph.edge.irreps,
            irreps_out=node_node_edge_tp_irreps_out,
            irrep_normalization=irrep_normalization,
        )

        self.layers.append(gate)

        # Intermediate layers
        if hidden_layers > 1:
            # Intermediate hidden
            for _ in range(hidden_layers - 1):
                self.layers.append(
                    _self_interaction(
                        self.layers[-1].irreps_out,
                        irreps_out=node_node_edge_tp_irreps_out,
                        irrep_normalization=irrep_normalization,
                    )
                )

        # Output layer
        self.layers.append(
            o3.TensorSquare(
                self.layers[-1].irreps_out,
                irreps_out=irreps_out,
                irrep_normalization=irrep_normalization,
            )
        )

        if rescaler is not None:
            self.layers.append(rescaler)

    @property
    def graph(self):
        return self._graph

    def forward(self, inputs) -> torch.Tensor:
        # Create the input representation
        out = self.node_node_tp(inputs["site1"], inputs["site2"])
        out = self.node_node_edge_tp(out, inputs["edge"])

        for layer in self.layers:
            out = layer(out)

        return out


def _self_interaction(
    irreps_in: o3.Irreps,
    irreps_out=None,
    filter_ir_out=None,
    irrep_normalization=None,
) -> Tuple:
    target_out = o3.TensorSquare(irreps_in, irreps_out, filter_ir_out=filter_ir_out).irreps_out

    scalars = o3.Irreps(filter(lambda mulir: mulir.ir.l == 0, target_out))
    # Now we know the target irreps out, we need to add scalars to the tensor product that can be used to gate the
    # non-scalars
    non_scalars = o3.Irreps(filter(lambda mulir: mulir.ir.l != 0, target_out))
    num_non_scalars = sum(mulir.mul for mulir in non_scalars)

    if num_non_scalars > 0:
        additional_scalars = num_non_scalars * o3.Irrep("0e")
        act = nn.Gate(
            irreps_scalars=scalars,
            act_scalars=[ACT_SCALARS[ir.p] for _, ir in scalars],
            irreps_gates=additional_scalars,
            act_gates=[ACT_NON_SCALARS[ir.p] for _, ir in additional_scalars],
            irreps_gated=non_scalars,
        )
        tp_irreps_out = additional_scalars + target_out
    else:
        additional_scalars = o3.Irreps()
        act = nn.Activation(scalars, [torch.nn.functional.silu])
        tp_irreps_out = target_out

    # Now we have the information we need to build the self-interaction
    tp = o3.TensorSquare(
        irreps_in,
        irreps_out=tp_irreps_out,
        filter_ir_out=filter_ir_out,
        irrep_normalization=irrep_normalization,
    )

    return tp, act


def _interaction(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    irrep_normalization=None,
):
    target_out = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out).irreps_out

    scalars = o3.Irreps(filter(lambda mulir: mulir.ir.l == 0, target_out))
    # Now we know the target irreps out, we need to add scalars to the tensor product that can be used to gate the
    # non-scalars
    non_scalars = o3.Irreps(filter(lambda mulir: mulir.ir.l != 0, target_out))
    num_non_scalars = sum(mulir.mul for mulir in non_scalars)

    if num_non_scalars > 0:
        additional_scalars = num_non_scalars * o3.Irrep("0e")
        act = nn.Gate(
            irreps_scalars=scalars,
            act_scalars=[ACT_SCALARS[ir.p] for _, ir in scalars],
            irreps_gates=additional_scalars,
            act_gates=[ACT_NON_SCALARS[ir.p] for _, ir in additional_scalars],
            irreps_gated=non_scalars,
        )
        tp_irreps_out = additional_scalars + target_out
    else:
        additional_scalars = o3.Irreps()
        act = nn.Activation(scalars, [torch.nn.functional.silu])
        tp_irreps_out = target_out

    # Now we have the information we need to build the self-interaction
    tp = o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, tp_irreps_out, irrep_normalization=irrep_normalization
    )
    return tp, act


ACT_SCALARS = {
    1: torch.nn.functional.silu,
    -1: torch.tanh,
}
ACT_NON_SCALARS = {
    1: torch.sigmoid,
    -1: torch.tanh,
}
