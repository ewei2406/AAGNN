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
from training import *
from models import GCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for model')
    parser.add_argument('--device', type=int, default=123,
                        help='Random seed for model')

    parser.add_argument('--data_dir', type=str, default='./tmp/',
                        help='Directory to download dataset')
    parser.add_argument('--data_seed', type=int, default=123,
                        help='Random seed for data split')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')

    parser.add_argument('--ptb_rate', type=float, default=1,
                        help='Perturbation rate')
    parser.add_argument('--reg_epochs', type=int, default=20,
                        help='Epochs to train models')
    parser.add_argument('--atk_epochs', type=int, default=20,
                        help='Epochs to attack data')

    parser.add_argument('--atk_lr', type=float, default=200,
                        help='Initial attack learning rate')
    parser.add_argument('--model_lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden_layers', type=int, default=32,
                        help='Number of hidden layers')

    parser.add_argument('--attack_type', type=str, default="min_min",  # min_max
                        help='minimize loss or maximize loss of adj matrix (min_min or min_max)')
    parser.add_argument('--train_parallel', type=str, default="Y",  # N
                        help='Train the surrogate model in parallel with the adj matrix perturbations or beforehand')
    parser.add_argument('--pretrain', type=str, default="Y",  # N
                        help='Train the surrogate model in parallel with the adj matrix perturbations or beforehand')

    parser.add_argument('--reset_model', type=str, default="N",  # Y
                        help='reset the surrogate model each attack epoch')

    parser.add_argument('--atk_train_loops', type=int, default=1,
                        help='number of times to train the surrogate model each attack epoch')
    parser.add_argument('--atk_adj_loops', type=int, default=1,
                        help='number of times to attempt to perturb adj further each epoch')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("#" * 30)
    print(f"{args.attack_type} type attack {'not ' if args.train_parallel != 'Y' else ''}trained in parallel")
    print("#" * 30)

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

    adj, features, labels = process(data, device)

    idx_train = indices_to_binary(data.idx_train, features.shape[0])
    idx_val = indices_to_binary(data.idx_val, features.shape[0])
    idx_test = indices_to_binary(data.idx_test, features.shape[0])

    print('==== Dataset ====')
    print(f'adj shape: {list(adj.shape)}')
    print(f'feature shape: {list(features.shape)}')
    print(f'num labels: {labels.max().item()+1}')
    print(f'split seed: {args.data_seed}')
    print(
        f'train|val|test: {idx_train.sum()}|{idx_val.sum()}|{idx_test.sum()}')
    
    show_change_matrix(adj, adj, labels)


if __name__ == "__main__":
    main()
