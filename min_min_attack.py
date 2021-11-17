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
    parser.add_argument('--gnn_path', type=str,
                        required=True, help='Path of saved model')
    # ['minmax', 'Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train', 'random']
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


if __name__ == "__main__":
    main()