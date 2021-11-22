import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import pickle as pkl
import copy
import networkx as nx

from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset

from deeprobust.graph.global_attack import MinMax
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.global_attack import Random


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='cuda')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for model')

    parser.add_argument('--data_seed', type=int, default=123,
                        help='Random seed for data split')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')

    parser.add_argument('--model', type=str,
                        default='minmax', help='model variant')
    parser.add_argument('--loss_type', type=str,
                        default='CE', help='loss type')
    parser.add_argument('--att_lr', type=float, default=200,
                        help='Initial learning rate')
    parser.add_argument('--perturb_epochs', type=int,
                        default=200, help='Number of epochs to poisoning loop')
    parser.add_argument('--ptb_rate', type=float,
                        default=0.05, help='pertubation rate')
    parser.add_argument('--loss_weight', type=float,
                        default=1.0, help='loss weight')
    parser.add_argument('--reg_weight', type=float,
                        default=0.0, help='regularization weight')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--data_dir', type=str, default='./tmp/',
                        help='Directory to download dataset')
    parser.add_argument('--target_node', type=str,
                        default='train', help='target node set')
    parser.add_argument('--sanitycheck', type=str, default='no',
                        help='whether store the intermediate results')

    parser.add_argument('--distance_type', type=str,
                        default='l2', help='distance type')
    parser.add_argument('--opt_type', type=str,
                        default='max', help='optimization type')
    parser.add_argument('--sample_type', type=str,
                        default='sample', help='sample type')

    args = parser.parse_args()

    args.device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Setting seeds ============================

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    # Envrionment ============================

    print('==== Environment ====')
    print(f'torch version: {torch.__version__}')
    print(f'device: {args.device}')
    print(f'torch seed: {args.seed}')

    # Load data ============================

    data = Dataset(root=args.data_dir, name=args.dataset,
                   setting='gcn', seed=args.data_seed)

    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    idx_unlabeled = np.union1d(idx_val, idx_test)

    print('==== Dataset ====')
    print(f'density: {nx.density(nx.from_numpy_array(adj))}')
    print(f'adj shape: {adj.shape}')
    print(f'feature shape: {features.shape}')
    print(f'label number: {labels.max().item()+1}')
    print(f'split seed: {args.data_seed}')
    print(
        f'train|valid|test set: {idx_train.shape}|{idx_val.shape}|{idx_test.shape}')

    # Create model ======================

    reg_model = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item()+1,
        nhid=args.hidden,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        device=args.device)
    reg_model = reg_model.to(args.device)

    # Train regularly ========================

    reg_model.fit(features, adj, labels, idx_train,
                  verbose=True, train_iters=20)

    # Evaluate performance ===================

    reg_model.test(idx_test)

    print('finished!')


if __name__ == "__main__":
    main()
