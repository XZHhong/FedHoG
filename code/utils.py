from typing import Any, Union

import torch
import dgl
from torch_geometric.data import Data, HeteroData


def from_dgl(g: Any,) -> Data:
    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)
        data.edge_weight = g.edata['weight']

        return data
