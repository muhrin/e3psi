import abc
from typing import Union

from e3nn import nn
from e3nn import o3
import torch

from . import graphs

__all__ = "OnsiteModel", "IntersiteModel", "Model"


class Model(torch.nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def graph(self) -> graphs.AbstractObj:
        """Get the graph that described this model"""

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class OnsiteModel(Model):
    """Model for predicting onsite values i.e. a node with self interactions"""

    def __init__(
        self,
        graph: graphs.AbstractObj,
        nn_irreps_out: Union[str, o3.Irreps] = None,
        irreps_out="0e",
    ):
        super().__init__()

        self._graph = graph
        self.irreps_in = self._graph.irreps

        node_node_tp_irreps_out = (
            o3.Irreps(nn_irreps_out)
            if nn_irreps_out is not None
            else o3.ReducedTensorProducts("ij=ji", i=self.irreps_in).irreps_out
        )

        self.gate = _gate_this(node_node_tp_irreps_out)
        # self.gate = nn.Gate(
        #     irreps_scalars="10x0e",
        #     act_scalars=[torch.nn.functional.silu],
        #     irreps_gates="13x0e",  #
        #     act_gates=[torch.sigmoid],
        #     irreps_gated="4x2e + 1x3e + 4x4e + 1x5e + 2x6e + 1x8e",
        # )

        self.tp1 = o3.TensorSquare(
            self.irreps_in, irreps_out=self.gate.irreps_in, irrep_normalization="norm"
        )  # contains the parameters of the network
        self.tp_out = o3.TensorSquare(
            self.gate.irreps_out, irreps_out=irreps_out, irrep_normalization="norm"
        )

    @property
    def graph(self):
        return self._graph

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = data["site"]

        out = self.tp1(out)
        out = self.gate(out)

        out = self.tp_out(out)

        return out


class IntersiteModel(Model):
    """Model for predicting intersite values, i.e. a two-body property"""

    def __init__(
        self,
        graph: graphs.TwoSite,
        n1n2_irreps_out=None,
        n1n2e_irreps_out=None,
        irreps_out="0e",
    ):
        super().__init__()

        self._graph = graph

        # Node-node TP
        node_node_tp_irreps_out = (
            o3.Irreps(n1n2_irreps_out)
            if n1n2_irreps_out is not None
            else o3.FullTensorProduct(
                self.graph.attrs.site1.irreps, self.graph.attrs.site2.irreps
            ).irreps_out
        )

        self.node_node_tp = o3.FullyConnectedTensorProduct(
            self.graph.site1.irreps, self.graph.site2.irreps, irreps_out=node_node_tp_irreps_out
        )

        # Node-node output * edge TP
        node_node_edge_tp_irreps_out = (
            o3.Irreps(n1n2e_irreps_out)
            if n1n2e_irreps_out is not None
            else o3.FullTensorProduct(
                self.node_node_tp.irreps_out, self.graph.attrs.edge.irreps
            ).irreps_out
        )

        self.node_node_edge_tp = o3.FullyConnectedTensorProduct(
            node_node_tp_irreps_out,
            self.graph.edge.irreps,
            irreps_out=node_node_edge_tp_irreps_out,
        )

        # Gate
        gate = _gate_this(self.node_node_edge_tp.irreps_out)
        self.gate = gate

        self.out_tp = o3.TensorSquare(
            gate.irreps_out, irreps_out=irreps_out, irrep_normalization="component"
        )

    @property
    def graph(self):
        return self._graph

    def forward(self, data) -> torch.Tensor:
        out = data

        out = self.node_node_tp(out["site1"], out["site2"])
        out = self.node_node_edge_tp(out, data["edge"])
        out = self.gate(out)

        out = self.out_tp(out)

        return out


def _gate_this(irreps_in: o3.Irreps, scalar_split=1.0) -> nn.Gate:
    total_scalars = sum(mulir.mul for mulir in irreps_in if mulir.ir.l == 0)
    non_scalars = o3.Irreps(list(mulir for mulir in irreps_in if mulir.ir.l != 0))
    num_non_scalars = sum(mulir.mul for mulir in non_scalars)
    num_scalars = total_scalars - num_non_scalars

    if num_scalars <= 0:
        raise ValueError(f"Don't have enough scalars to gate all non-scalar irreps: {irreps_in}")

    return nn.Gate(
        irreps_scalars=num_scalars * o3.Irrep("0e"),
        act_scalars=[torch.nn.functional.silu],
        irreps_gates=num_non_scalars * o3.Irrep("0e"),
        act_gates=[torch.sigmoid],
        irreps_gated=non_scalars,
    )
