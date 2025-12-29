'''
This file is used to define classes and functions related to gnn
'''
from typing import Optional, Union
import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data


class LightGCN(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, embs: Tensor, edge_index: Adj, edge_weight: OptTensor = None,) -> Tensor:
        out = embs * self.alpha[0]

        for i in range(self.num_layers):
            embs = self.convs[i](embs, edge_index, edge_weight)
            out = out + embs * self.alpha[i + 1]

        return out


class LightGCN_2(torch.nn.Module):
    def __init__(self, adj_mat: Data, num_layers: int, num_rows: int, num_cols: int):
        self.num_layers = num_layers

        # convert pyg to pytorch
        adj_mat = torch.sparse_coo_tensor(
            adj_mat.edge_index.flip(dims=[0]), adj_mat.edge_weight, (num_rows, num_cols)
        )

        # construct degree matrix
        deg_mat = torch.sparse.sum(adj_mat, dim=1)
        deg_mat = torch.sparse_coo_tensor(
            torch.vstack((deg_mat.indices()[0], deg_mat.indices()[0])), deg_mat.values(), (num_rows, num_cols)
        )

        # normalize the adjacency matrix
        self.nor_adj_mat = torch.sparse.mm(
            torch.sparse.mm(deg_mat.sqrt().pow(-1), adj_mat),
            deg_mat.sqrt().pow(-1)
        )

    def get_final_embedding(self, embs):
        emb_list = [embs]
        for _ in range(self.num_layers):
            emb_list.append(torch.sparse.mm(self.nor_adj_mat, emb_list[-1]))

        final_embs = torch.stack(emb_list, dim=2).mean(dim=2)

        return final_embs
