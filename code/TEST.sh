# gowalla
CUDA_VISIBLE_DEVICES=0 python main.py --emb_size=64 --lr=0.001 --num_chosen_users=128 --graph_topk=10 --num_uu_layers=3 --num_ii_layers=3 --reg=1e-2 --dataset_name="gowalla"
# yelp2018
CUDA_VISIBLE_DEVICES=0 python main.py --emb_size=64 --lr=0.001 --num_chosen_users=128 --graph_topk=20 --num_uu_layers=3 --num_ii_layers=3 --reg=1e-2 --dataset_name="yelp2018"
# amazon-book
CUDA_VISIBLE_DEVICES=0 python main.py --emb_size=64 --lr=0.001 --num_chosen_users=128 --graph_topk=10 --num_uu_layers=3 --num_ii_layers=3 --reg=1e-2 --dataset_name="amazon-book"
