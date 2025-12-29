'''
This file is used to define classes and functions related to server
'''
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pygu
from torch_geometric.nn import GCN, GAT, GraphSAGE
import dgl
import time
from dataset import Dataset
from gnn import LightGCN
from clients import Clients
from log import Log
import utils


class Servers(object):
    def __init__(self, args, dataset: Dataset):
        self.args = args
        self.dataset = dataset

        self.training_set = dataset.training_set
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.emb_size = args.emb_size
        self.num_uu_layers = args.num_uu_layers
        self.num_ii_layers = args.num_ii_layers
        self.graph_topk = args.graph_topk
        self.preprocess_way = args.preprocess_way
        self.gnn_type = args.gnn_type
        self.reg = args.reg
        self.num_epochs = args.num_epochs
        self.top_K = args.top_K
        self.early_stopping = args.early_stopping
        self.device = args.device
        self.dropout = args.dropout

        # initialize server and third-party server
        self.u_embs = nn.Embedding(self.num_users, self.emb_size, device=self.device)
        self.i_embs = nn.Embedding(self.num_items, self.emb_size, device=self.device)
        nn.init.xavier_uniform_(self.u_embs.weight)
        nn.init.xavier_uniform_(self.i_embs.weight)

        self.uu_graph, self.ii_graph = self.construct_graph()
        self.uu_gnn, self.ii_gnn = self.init_gnn()

        # initialize clients
        self.clients = Clients(args, dataset)

        # optimizer
        self.optimizer = torch.optim.AdamW([{'params': self.u_embs.parameters()}, {'params': self.i_embs.parameters()},
                                            {'params': self.uu_gnn.parameters()}, {'params': self.ii_gnn.parameters()}], lr=args.lr, weight_decay=args.reg)

        # log
        self.log = Log(args)

        # num_pos
        num_pos = []
        for u_id in range(self.num_users):
            num_pos.append(len(self.training_set[u_id]))
        self.num_pos = torch.tensor(num_pos).reshape(-1, 1).to(self.device)

    def construct_graph(self):
        # construct the user-item interaction adjacency matrix, which doesn't actually exist in paper's setting, just for simplifying the code
        ui_adj_mat_indices = [[], []]
        for u_id, interacted_i_ids in self.training_set.items():
            ui_adj_mat_indices[0] += [u_id for i in range(len(interacted_i_ids))]
            ui_adj_mat_indices[1] += interacted_i_ids

        ui_adj_mat = torch.sparse_coo_tensor(
            ui_adj_mat_indices, [1 for i in range(len(ui_adj_mat_indices[0]))], (self.num_users, self.num_items)
        ).to(torch.float)

        # construct degree matrices w.r.t. the user-item interaction adjacency matrix
        ui_ude_mat = torch.sparse.sum(ui_adj_mat, dim=1)
        ui_ude_mat = torch.sparse_coo_tensor(
            torch.vstack((ui_ude_mat.indices()[0], ui_ude_mat.indices()[0])), ui_ude_mat.values(), (self.num_users, self.num_users)
        )

        ui_ide_mat = torch.sparse.sum(ui_adj_mat, dim=0)
        ui_ide_mat = torch.sparse_coo_tensor(
            torch.vstack((ui_ide_mat.indices()[0], ui_ide_mat.indices()[0])), ui_ide_mat.values(), (self.num_items, self.num_items)
        )

        # construct the user-user co-interacted matrix and item-item co-interacted matrix
        # omit the real construction process proposed in the paper for simplifying the code
        uu_adj_mat = torch.sparse.mm(ui_adj_mat, ui_adj_mat.transpose(0, 1))
        # print(ui_adj_mat)
        # print(uu_adj_mat)
        # print(uu_adj_mat.coalesce().values().sum()/uu_adj_mat._nnz())
        ii_adj_mat = torch.sparse.mm(ui_adj_mat.transpose(0, 1), ui_adj_mat)

        # preprocess the user-user and item-item matrix
        if self.preprocess_way == 1:
            # let the elements E_ui be the cosine similarity of the R_u and R_i, which R is the interaction matrix
            uu_adj_mat = torch.sparse.mm(
                torch.sparse.mm(ui_ude_mat.sqrt().pow(-1), uu_adj_mat),
                ui_ude_mat.sqrt().pow(-1)
            )
            ii_adj_mat = torch.sparse.mm(
                torch.sparse.mm(ui_ide_mat.sqrt().pow(-1), ii_adj_mat),
                ui_ide_mat.sqrt().pow(-1)
            )

            # remove self-loops
            uu_graph = dgl.graph((uu_adj_mat.coalesce().indices()[0], uu_adj_mat.coalesce().indices()[1]))
            uu_graph.edata['weight'] = uu_adj_mat.coalesce().values()
            uu_graph = dgl.remove_self_loop(uu_graph)
            ii_graph = dgl.graph((ii_adj_mat.coalesce().indices()[0], ii_adj_mat.coalesce().indices()[1]))
            ii_graph.edata['weight'] = ii_adj_mat.coalesce().values()
            ii_graph = dgl.remove_self_loop(ii_graph)

            # keep topk largest neighbors of each node(each col of the matrix)
            uu_graph = dgl.sampling.select_topk(uu_graph, self.graph_topk, 'weight')
            ii_graph = dgl.sampling.select_topk(ii_graph, self.graph_topk, 'weight')

            # convert dgl to pyg
            uu_graph = utils.from_dgl(uu_graph).to(self.device)
            ii_graph = utils.from_dgl(ii_graph).to(self.device)

        return uu_graph, ii_graph

    def init_gnn(self):
        if self.gnn_type == 'LightGCN':
            uu_gnn = LightGCN(self.num_uu_layers).to(self.device)
            ii_gnn = LightGCN(self.num_ii_layers).to(self.device)
        elif self.gnn_type == 'GCN':
            uu_gnn = GCN(in_channels=self.emb_size, hidden_channels=self.emb_size,
                         num_layers=self.num_uu_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)
            ii_gnn = GCN(in_channels=self.emb_size, hidden_channels=self.emb_size,
                         num_layers=self.num_ii_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)
        elif self.gnn_type == 'GAT':
            uu_gnn = GAT(in_channels=self.emb_size, hidden_channels=self.emb_size,
                         num_layers=self.num_uu_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)
            ii_gnn = GAT(in_channels=self.emb_size, hidden_channels=self.emb_size,
                         num_layers=self.num_ii_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)
        elif self.gnn_type == 'GraphSAGE':
            uu_gnn = GraphSAGE(in_channels=self.emb_size, hidden_channels=self.emb_size,
                               num_layers=self.num_uu_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)
            ii_gnn = GraphSAGE(in_channels=self.emb_size, hidden_channels=self.emb_size,
                               num_layers=self.num_ii_layers, out_channels=self.emb_size, dropout=self.dropout).to(self.device)

        return uu_gnn, ii_gnn

    def train_and_test(self):
        metrics_per_epoch, best_epoch, best_rec, stop_count = [], 0, 0, 0

        for epoch in range(self.num_epochs+1):
            # train
            if epoch != 0:
                if self.args.use_fast == 0:
                    loss = self.train()
                else:
                    loss = self.faster_train()
            else:
                loss = 0.0

            # test
            Val_Recall, Precision, Recall, NDCG = self.eval()
            output_info = (('epoch:%04d  loss: %.5f  ValRec: %.5f  Pre: %.5f  Rec: %.5f  NDCG: %.5f  time: ' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                           % (epoch, loss, Val_Recall, Precision, Recall, NDCG))
            print(output_info)
            self.log.info(output_info)
            metrics_per_epoch.append([Val_Recall, Precision, Recall, NDCG])

            # early stopping
            if Val_Recall >= best_rec:
                stop_count = 0
                best_rec = Val_Recall
                best_epoch = epoch
            else:
                stop_count += 1
                if stop_count >= self.early_stopping:
                    break

        output_info = 'The best epoch is %d, metrics: %s' % (best_epoch, str(metrics_per_epoch[best_epoch]))
        print(output_info)
        self.log.info(output_info)

    def train(self):
        self.uu_gnn.train()
        self.ii_gnn.train()

        # generate user(i.e., clients) batches
        u_batch_generator = self.dataset.generate_user_batches()
        batches_loss = []

        for u_batch in u_batch_generator:
            self.optimizer.zero_grad()

            # refine user and item embeddings
            u_refined_embs = self.uu_gnn(self.u_embs.weight, self.uu_graph.edge_index, edge_weight=self.uu_graph.edge_weight)
            i_refined_embs = self.ii_gnn(self.i_embs.weight, self.ii_graph.edge_index, edge_weight=self.ii_graph.edge_weight)

            # send refined embeddings to clients
            self.clients.receive_embs(u_refined_embs.detach().clone(), i_refined_embs.detach().clone())

            # clients training
            clients_grad_info, clients_loss = self.clients.train(u_batch)
            batches_loss.append(clients_loss)

            # FedAvg
            self.fedavg(clients_grad_info, u_refined_embs, i_refined_embs)

            # update servers' parameters
            self.optimizer.step()

        batches_loss = torch.mean(torch.hstack(batches_loss))

        return batches_loss

    def faster_train(self):
        # equivalent to self.train()

        self.uu_gnn.train()
        self.ii_gnn.train()

        # generate user(i.e., clients) batches
        u_batch_generator = self.dataset.generate_user_batches_for_fasttrain()
        batches_loss = []

        for batch_size, sample_nums, u_ids, pos_i_ids, neg_i_ids in u_batch_generator:
            self.optimizer.zero_grad()

            # refine user and item embeddings
            u_refined_embs = self.uu_gnn(self.u_embs.weight, self.uu_graph.edge_index, edge_weight=self.uu_graph.edge_weight)
            i_refined_embs = self.ii_gnn(self.i_embs.weight, self.ii_graph.edge_index, edge_weight=self.ii_graph.edge_weight)

            # backpropagation
            u_embs = u_refined_embs[u_ids]
            pos_i_embs = i_refined_embs[pos_i_ids]
            neg_i_embs = i_refined_embs[neg_i_ids]

            if self.args.score_function == 'dot_product':
                pos_ratings = torch.mul(u_embs, pos_i_embs).sum(dim=1)
                neg_ratings = torch.mul(u_embs, neg_i_embs).sum(dim=1)

            sample_nums = torch.tensor(sample_nums).to(self.device)
            bpr_loss = ((-F.logsigmoid(pos_ratings-neg_ratings))/sample_nums).mean()
            bpr_loss.backward()
            self.u_embs.weight.grad *= len(u_ids)
            self.i_embs.weight.grad *= (len(u_ids)/batch_size)
            batches_loss.append(bpr_loss.detach())

            # update servers' parameters
            self.optimizer.step()

        batches_loss = torch.mean(torch.hstack(batches_loss))

        return batches_loss

    def test_comp_time(self):
        self.uu_gnn.train()
        self.ii_gnn.train()

        # generate user(i.e., clients) batches
        u_batch_generator = self.dataset.generate_user_batches()
        batches_loss = []

        total_time = 0
        for u_batch in u_batch_generator:
            self.optimizer.zero_grad()

            # refine user and item embeddings
            u_refined_embs = self.uu_gnn(self.u_embs.weight, self.uu_graph.edge_index, edge_weight=self.uu_graph.edge_weight)
            i_refined_embs = self.ii_gnn(self.i_embs.weight, self.ii_graph.edge_index, edge_weight=self.ii_graph.edge_weight)

            # send refined embeddings to clients
            self.clients.receive_embs(u_refined_embs.detach().clone(), i_refined_embs.detach().clone())

            # clients training
            t1 = time.time()
            clients_grad_info, clients_loss = self.clients.train(u_batch)
            t2 = time.time()
            total_time += t2-t1
            batches_loss.append(clients_loss)

            # FedAvg
            self.fedavg(clients_grad_info, u_refined_embs, i_refined_embs)

            # update servers' parameters
            self.optimizer.step()

        output_info = 'Training time: %.5fs\n' % (total_time) + 'Training time per client: %.10fs\n' % (total_time/self.num_users)
        print(output_info)
        self.log.info(output_info)

    def fedavg(self, clients_grad_info, u_refined_embs: torch.Tensor, i_refined_embs: torch.Tensor):
        batch_size = len(clients_grad_info)

        u_grads = torch.zeros(u_refined_embs.shape, device=self.device)
        i_grads = torch.zeros(i_refined_embs.shape, device=self.device)
        for u_id, u_emb_grad, i_ids, i_emb_grads in clients_grad_info:
            u_grads[u_id] = u_emb_grad
            i_grads[i_ids] += i_emb_grads/batch_size

        u_refined_embs.backward(gradient=u_grads)
        i_refined_embs.backward(gradient=i_grads)

    @torch.no_grad()
    def eval(self):
        self.uu_gnn.eval()
        self.ii_gnn.eval()

        # refine user and item embeddings
        u_refined_embs = self.uu_gnn(self.u_embs.weight, self.uu_graph.edge_index, edge_weight=self.uu_graph.edge_weight)
        i_refined_embs = self.ii_gnn(self.i_embs.weight, self.ii_graph.edge_index, edge_weight=self.ii_graph.edge_weight)

        # send refined embeddings to clients
        self.clients.receive_embs(u_refined_embs.detach().clone(), i_refined_embs.detach().clone())

        # test clients
        metrics = self.clients.eval()

        return metrics
