'''
This file is used to define classes and functions related to datasets
'''
import os
import torch
import argparse
import numpy as np


class Dataset(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_chosen_users = args.num_chosen_users

        # load the dataset
        self.training_set, self.val_set, self.test_set, self.num_users, self.num_items = self.load_dataset(args)

    def load_dataset(self, args):
        dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
        training_set_path, val_set_path, test_set_path = os.path.join(dataset_dir, 'train.txt'), os.path.join(dataset_dir, 'val.txt'), os.path.join(dataset_dir, 'test.txt')
        training_set, val_set, test_set, num_users, num_items = {}, {}, {}, float('-inf'), float('-inf')

        # load the training set, training_set[u] = [items that user u has interacted with in the training set]
        with open(training_set_path) as f:
            for line in f.readlines():
                line = list(map(int, line.strip().split()))
                num_users, num_items = max(num_users, line[0]+1), max(num_items, max(line[1:])+1)
                training_set[line[0]] = line[1:]

        # load the validation set, val_set[u] = [items that user u has interacted with in the validation set]
        with open(val_set_path) as f:
            for line in f.readlines():
                line = list(map(int, line.strip().split()))
                num_users, num_items = max(num_users, line[0]+1), max(num_items, max(line[1:])+1)
                val_set[line[0]] = line[1:]

        # load the test set, test_set[u] = [items that user u has interacted with in the test set]
        with open(test_set_path) as f:
            for line in f.readlines():
                line = list(map(int, line.strip().split()))
                num_users, num_items = max(num_users, line[0]+1), max(num_items, max(line[1:])+1 if len(line[1:]) > 0 else float('-inf'))
                if len(line[1:]) > 0:
                    test_set[line[0]] = line[1:]

        return training_set, val_set, test_set, num_users, num_items

    def generate_user_batches(self):
        u_ids = np.arange(0, self.num_users)
        np.random.shuffle(u_ids)
        u_ids_list = np.split(u_ids, np.arange(self.num_chosen_users, len(u_ids), self.num_chosen_users))

        for u_ids in u_ids_list:
            pos_i_ids_list, neg_i_ids_list = self.sample_items(u_ids)
            yield zip(u_ids, pos_i_ids_list, neg_i_ids_list)

    def sample_items(self, u_ids: np.ndarray):
        pos_i_ids_list, neg_i_ids_list = [], []
        for u_id in u_ids:
            num_sample_items = len(self.training_set[u_id])

            # sample positive items
            interacted_i_ids = self.training_set[u_id]
            pos_i_ids = np.random.choice(interacted_i_ids, num_sample_items)

            # sample negative items
            neg_i_ids = np.random.randint(0, self.num_items, num_sample_items)
            while (isin := np.isin(neg_i_ids, interacted_i_ids)).sum() > 0:
                neg_i_ids[isin] = np.random.randint(0, self.num_items, isin.sum())

            pos_i_ids_list.append(pos_i_ids)
            neg_i_ids_list.append(neg_i_ids)

        return pos_i_ids_list, neg_i_ids_list

    def generate_user_batches_for_fasttrain(self):
        u_ids = np.arange(0, self.num_users)
        np.random.shuffle(u_ids)
        u_ids_list = np.split(u_ids, np.arange(self.num_chosen_users, len(u_ids), self.num_chosen_users))

        for u_ids in u_ids_list:
            batch_size, sample_nums, batch_u_ids, batch_pos_i_ids, batch_neg_i_ids = len(u_ids), [], [], [], []
            for u_id in u_ids:
                num_sample_items = len(self.training_set[u_id])

                # user id and sample number
                batch_u_ids += [u_id]*num_sample_items
                sample_nums += [num_sample_items]*num_sample_items

                # sample positive items
                interacted_i_ids = self.training_set[u_id]
                pos_i_ids = np.random.choice(interacted_i_ids, num_sample_items)

                # sample negative items
                neg_i_ids = np.random.randint(0, self.num_items, num_sample_items)
                while (isin := np.isin(neg_i_ids, interacted_i_ids)).sum() > 0:
                    neg_i_ids[isin] = np.random.randint(0, self.num_items, isin.sum())

                batch_pos_i_ids.append(pos_i_ids)
                batch_neg_i_ids.append(neg_i_ids)

            batch_pos_i_ids, batch_neg_i_ids = np.hstack(batch_pos_i_ids), np.hstack(batch_neg_i_ids)

            yield batch_size, sample_nums, batch_u_ids, batch_pos_i_ids, batch_neg_i_ids
