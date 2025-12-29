'''
This file is used to define classes and functions related to clients
'''
import os
import time
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Iterable
from dataset import Dataset


class Clients(object):
    def __init__(self, args, dataset: Dataset):
        self.args = args
        self.dataset = dataset

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.score_function = args.score_function
        self.reg = args.reg
        self.top_K = args.top_K
        self.early_stopping = args.early_stopping
        self.device = args.device

        # construct excluded rows and columns when evaluating
        self.exclude_row_index = []
        self.exclude_col_index = []
        for u_id, interacted_i_ids in dataset.training_set.items():
            self.exclude_row_index += [u_id]*len(interacted_i_ids)
            self.exclude_col_index += interacted_i_ids

    def receive_embs(self, u_embs: torch.Tensor, i_embs: torch.Tensor):
        self.u_embs = u_embs
        self.u_embs.requires_grad_()
        self.i_embs = i_embs
        self.i_embs.requires_grad_()

    def train(self, u_batch):
        # each client train itself
        grad_info, batch_loss = [], []
        for u_id, pos_i_ids, neg_i_ids in u_batch:
            i_ids = np.concatenate((np.unique(pos_i_ids), np.unique(neg_i_ids)))

            # calculate BPR Loss
            loss = self.cal_bprloss(u_id, pos_i_ids, neg_i_ids)
            loss.backward()
            batch_loss.append(loss.detach())

            # get gradients w.r.t. embeddings
            u_emb_grad = self.u_embs.grad[u_id].detach().clone()
            i_emb_grads = self.i_embs.grad[i_ids].detach().clone()
            self.u_embs.grad.zero_()
            self.i_embs.grad.zero_()
            grad_info.append([u_id, u_emb_grad, i_ids, i_emb_grads])

        batch_loss = torch.mean(torch.hstack(batch_loss))

        return grad_info, batch_loss

    def cal_bprloss(self, u_id, pos_i_ids, neg_i_ids):
        u_emb = self.u_embs[u_id]
        pos_i_embs = self.i_embs[pos_i_ids]
        neg_i_embs = self.i_embs[neg_i_ids]

        # calculate ratings
        if self.score_function == 'dot_product':
            pos_ratings = torch.mul(u_emb, pos_i_embs).sum(dim=1)
            neg_ratings = torch.mul(u_emb, neg_i_embs).sum(dim=1)

        # calculate bpr loss
        bpr_loss = (-F.logsigmoid(pos_ratings-neg_ratings)).mean()
        # reg_loss = torch.cat((u_emb.unsqueeze(0),  pos_i_embs, neg_i_embs),
        #                      0).pow(2).sum()/len(pos_ratings)*self.reg*(1/2)
        # loss = bpr_loss+reg_loss
        loss = bpr_loss

        return loss

    @torch.no_grad()
    def eval(self):
        # divide batches to prevent OOM
        topk_i_ids_list = []
        for u_ids in np.array_split([i for i in range(self.num_users)], 200):
            # predict
            if self.score_function == 'dot_product':
                rating_mat = (self.u_embs[u_ids]@self.i_embs.T)

            # exclude interaction items in the training set
            exclude_row_index, exclude_col_index = [], []
            for index, u_id in enumerate(u_ids):
                interacted_i_ids = self.dataset.training_set[u_id]
                exclude_row_index += [index]*len(interacted_i_ids)
                exclude_col_index += interacted_i_ids
            rating_mat[exclude_row_index, exclude_col_index] = float('-inf')

            # get top-K item list
            _, topk_i_ids = torch.topk(rating_mat, k=self.top_K)
            topk_i_ids_list.append(topk_i_ids.cpu().numpy())
        topk_i_ids_list = np.vstack(topk_i_ids_list)

        # calculate metrics of validation set
        Val_Recall = 0
        for u_id in list(self.dataset.val_set.keys()):
            topk_i_ids, val_i_ids = topk_i_ids_list[u_id], self.dataset.val_set[u_id]

            # whether the recommended items is in validation set
            isin_valset = np.isin(topk_i_ids, val_i_ids)

            # calculate metrics
            Recall = isin_valset.sum()/len(val_i_ids)
            Val_Recall += Recall
        Val_Recall /= len(self.dataset.val_set)

        # calculate metrics of test set
        metrics = []
        DCG_denominator = np.log2([i+1 for i in range(1, self.top_K+1)])
        NDCG_denominator = (1/DCG_denominator).sum()
        for u_id in list(self.dataset.test_set.keys()):
            topk_i_ids, test_i_ids = topk_i_ids_list[u_id], self.dataset.test_set[u_id]

            # whether the recommended items is in test set
            isin_testset = np.isin(topk_i_ids, test_i_ids)

            # calculate metrics
            Precision = isin_testset.sum()/self.top_K
            Recall = isin_testset.sum()/len(test_i_ids)
            if (len_test := len(test_i_ids)) >= self.top_K:
                NDCG = (isin_testset/DCG_denominator).sum()/NDCG_denominator
            else:
                denominator = (1/np.log2([i+1 for i in range(1, len_test+1)])).sum()
                NDCG = (isin_testset/DCG_denominator).sum()/denominator

            metrics.append([Precision, Recall, NDCG])

        metrics = np.array(metrics).mean(axis=0)

        return np.hstack((Val_Recall, metrics))
