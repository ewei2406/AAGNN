import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

from deeprobust.graph.data import Dataset

from utils import *
from models import GCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for model')
    parser.add_argument('--device', type=int, default=123,
                        help='Random seed for model')
    parser.add_argument('--ptb_rate', type=float, default=0.1,
                        help='Perturbation rate')
    parser.add_argument('--reg_epochs', type=int, default=10,
                        help='Epochs to train models')
    parser.add_argument('--atk_epochs', type=int, default=10,
                        help='Epochs to attack data')

    parser.add_argument('--data_dir', type=str, default='./tmp/',
                        help='Directory to download dataset')
    parser.add_argument('--data_seed', type=int, default=123,
                        help='Random seed for data split')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################################################
    # Setup environment
    ################################################

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    print('==== Environment ====')
    print(f'torch version: {torch.__version__}')
    print(f'device: {device}')
    print(f'torch seed: {args.seed}')

    ################################################
    # Load data
    ################################################

    data = Dataset(root=args.data_dir, name=args.dataset,
                   setting='gcn', seed=args.data_seed)

    adj, features, labels = csr_to_tensor(data.adj).int(), csr_to_tensor(data.features).float(), torch.from_numpy(data.labels)

    idx_train = indices_to_binary(data.idx_train, features.shape[0])
    idx_val = indices_to_binary(data.idx_val, features.shape[0])
    idx_test = indices_to_binary(data.idx_test, features.shape[0])

    # print(adj, features, labels)
    print(adj[0], features[0], labels[0])
    print(idx_train[0], idx_val[0], idx_test[0])


    dataset = Planetoid(root='./tmp/Cora', name='Cora')
    data = dataset[0].to(device)

    features, adj, labels = data.x, to_adj(data.edge_index), data.y
    idx_train, idx_test, idx_val = data.train_mask, data.test_mask, data.val_mask

    # print(adj, features, labels)
    print(adj[0], features[0], labels[0])
    print(idx_train[0], idx_val[0], idx_test[0])



    print('==== Dataset ====')
    # print(f'density: {nx.density(nx.from_numpy_array(adj))}')
    print(f'adj shape: {list(adj.shape)}')
    print(f'feature shape: {list(features.shape)}')
    print(f'num labels: {labels.max().item()+1}')
    # print(f'split seed: {args.data_seed}')
    print(
        f'train|val|test: {idx_train.sum()}|{idx_val.sum()}|{idx_test.sum()}')


if __name__ == "__main__":
    main()
    print("finished!")
