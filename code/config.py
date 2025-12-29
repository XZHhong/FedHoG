'''
This file is used to configure parameters
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--emb_size', type=int, default=64,
                        help='The embedding size ')
    parser.add_argument('--num_uu_layers', type=int, default=2,
                        help="The number of model's layers w.r.t. user-user graph")
    parser.add_argument('--num_ii_layers', type=int, default=2,
                        help="The number of model's layers w.r.t. item-item graph")
    parser.add_argument('--preprocess_way', type=int, default=1,
                        help="The way of preprocessing the user-user and item-item matrix")
    parser.add_argument('--graph_topk', type=int, default=10,
                        help="Choose topk neighbors of the user-user and item-item matrix")
    parser.add_argument('--gnn_type', type=str, default='LightGCN', choices=['LightGCN', 'GCN', 'GAT', 'GraphSAGE'],
                        help="Choose a gnn from ['LightGCN', 'GCN', 'GAT', 'GraphSAGE']")
    parser.add_argument('--score_function', type=str, default='dot_product', choices=['dot_product'],
                        help="Choose a score function from [dot_product]")

    # Dataset settings
    parser.add_argument('--dataset_name', type=str, default='gowalla', choices=['gowalla', 'yelp2018', 'amazon-book'],
                        help='Choose a dataset from [gowalla, yelp2018, amazon-book]')
    parser.add_argument('--dataset_dir', type=str, default='../data',
                        help='Folder path to store datasets')

    # Experimental environment settings
    parser.add_argument('--seed', type=int, default=2023,
                        help='Set the random seed to facilitate reproduction')
    parser.add_argument('--use_seed', type=int, default=1,
                        help='Whether to use the random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Choose a device from [cpu, cuda]')
    parser.add_argument('--save_dir', type=str, default='../result',
                        help='Folder path to store training and test results')
    parser.add_argument('--save_name', type=str, default='None',
                        help='Folder name to store training and test results')
    parser.add_argument('--use_fast', type=int, default=1,
                        help='Whether to use the fast training')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='The number of epochs to train')
    parser.add_argument('--num_chosen_users', type=int, default=128,
                        help='The number of users chosen in each round')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate')
    parser.add_argument('--reg', type=float, default=1e-2,
                        help='The coefficient of L2 regularization')
    parser.add_argument('--early_stopping', type=int, default=50,
                        help='The number of epochs to wait before early stop')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout for GNN')

    # Eval settings
    parser.add_argument('--top_K', type=int, default=20,
                        help='top K')

    return parser.parse_args()
