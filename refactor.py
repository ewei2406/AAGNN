import torch
import os
import argparse
import numpy as np

from aagnn.GCN import GCN
from aagnn.graphData import Graph

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

    parser.add_argument('--model_lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden_layers', type=int, default=32,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for GCN')

    parser.add_argument('--attack_type', type=str, default="min_max",
                        help='minimize loss or maximize loss of adj matrix (min_min or min_max)')
    parser.add_argument('--protected_label', type=int, default=3, 
                        help='label to "protect"')
    parser.add_argument('--ptb_rate', type=float, default=0.5,
                        help='Perturbation rate (percentage of available edges)')

    parser.add_argument('--reg_epochs', type=int, default=10,
                        help='Epochs to train models')
    parser.add_argument('--ptb_epochs', type=int, default=5,
                        help='Epochs to perturb adj matrix')
    parser.add_argument('--surrogate_epochs', type=int, default=10,
                        help='Epochs to train surrogate')
    parser.add_argument('--surrogate_train', type=str, default="N", #Y
                        help='Enable continual training on surrogate')
    
    parser.add_argument('--csv', type=str, default='',
                        help='save the outputs to csv')

    parser.add_argument('--edge_case', type=str, default='', # add, remove
                        help='run edge cases')

    ################################################
    # Setup environment
    ################################################

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    graph = Graph(args.data_dir, args.dataset, 'gcn', args.data_seed, device)
    graph.summarize()

    print(graph.idx_train.nonzero())
    
    ################################################
    # Baseline
    ################################################

    baseline = GCN(
        input_features=graph.features.shape[1],
        output_classes=graph.labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name="baseline"
        )

    baseline.fit(graph, 10)

    ################################################
    # Baseline
    ################################################

if __name__ == "__main__":
    print()
    main()
