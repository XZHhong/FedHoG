import numpy as np
import torch
import config
from servers import Servers
from dataset import Dataset


if __name__ == '__main__':
    # read configuration
    args = config.parse_args()

    # set random seed
    if args.use_seed:
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)

    # load dataset
    dataset = Dataset(args)

    # initialize server and clients
    servers = Servers(args, dataset)

    # train and test
    servers.train_and_test()
