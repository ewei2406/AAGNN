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


    parser.add_argument('--atk_lr', type=float, default=200,
                        help='Initial attack learning rate')
    parser.add_argument('--model_lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden_layers', type=int, default=32,
                        help='Number of hidden layers')

    parser.add_argument('--ptb_rate', type=float, default=0.05,
                        help='Perturbation rate (percentage of available edges)')
    parser.add_argument('--reg_epochs', type=int, default=20,
                        help='Epochs to train models')
    parser.add_argument('--atk_epochs', type=int, default=10,
                        help='Epochs to attack data')

    parser.add_argument('--attack_type', type=str, default="min_min",  # min_max
                        help='minimize loss or maximize loss of adj matrix (min_min or min_max)')
    parser.add_argument('--reset_model', type=str, default="N",  # Y
                        help='reset the surrogate model each attack epoch')
    parser.add_argument('--pretrain_epochs', type=int, default=5,
                        help='number of times train the surrogate before attacking')

    parser.add_argument('--atk_train_loops', type=int, default=0,
                        help='number of times to train the surrogate model each attack epoch')
    parser.add_argument('--atk_adj_loops', type=int, default=1,
                        help='number of times to attempt to perturb adj further each epoch')

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

    ################################################
    # Train a model regularly
    ################################################

    print('==== Training regular model ====')

    model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers
    ).to(device)

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    for epoch in range(args.reg_epochs):
        predictions = train_step(
            model=model,
            optimizer=optimizer,
            features=features,
            adj=adj,
            labels=labels,
            idx_train=idx_train,
            loss_fn=F.cross_entropy
        )

        show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

    ################################################
    # Attack the data
    ################################################

    num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

    surrogate_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers
    ).to(device)

    optimizer = torch.optim.Adam(
        surrogate_model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    perturbations = torch.zeros_like(adj).float()

    perturbations.requires_grad = True

    # Pretrain the surrogate model ============

    print("==== Pretraining surrogate ====")

    for epoch in range(args.pretrain_epochs):
        predictions = train_step(
            model=surrogate_model,
            optimizer=optimizer,
            features=features,
            adj=adj,
            labels=labels,
            idx_train=idx_train,
            loss_fn=F.cross_entropy
        )

        show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

    # Attack the data =========================

    print('==== Attacking Data ====')

    for epoch in range(args.atk_epochs):

        # Reset the model if desired ==========

        if args.reset_model == "Y":

            print("Surrogate reset")

            surrogate_model = GCN(
                input_features=features.shape[1],
                output_classes=labels.max().item()+1,
                hidden_layers=args.hidden_layers
            ).to(device)

            optimizer = torch.optim.Adam(surrogate_model.parameters(),
                                         lr=args.model_lr, weight_decay=args.weight_decay)


        # Train the model ====================

        print(f'== Training surrogate ({args.atk_train_loops}) ==')

        surrogate_model.train()
        # modified_adj = normalize_adj(modified_adj) # Normalize

        for i in range(args.atk_train_loops):
            modified_adj = get_modified_adj(adj, perturbations)

            predictions = train_step(
                model=surrogate_model,
                optimizer=optimizer,
                features=features,
                adj=modified_adj,
                labels=labels,
                idx_train=idx_train,
                loss_fn=F.cross_entropy
            )

        show_acc(epoch+1, predictions, labels,
                    idx_train, idx_test, False)

        # Perturb the adj matrix =================

        print(f'== Perturbing adj ({args.atk_adj_loops}) ==')

        surrogate_model.eval()

        for i in range(args.atk_adj_loops):

            perturbations = adj_step(
                surrogate_model=surrogate_model,
                features=features,
                adj=adj,
                labels=labels,
                idx_train=idx_train,
                loss_fn=F.cross_entropy,
                loss_inverse=args.attack_type == "min_min",
                perturbations=perturbations,
                epoch=epoch,
                lr_0=args.atk_lr,
                num_perturbations=num_perturbations
            )

    # Draw samples from best ==========

    best = random_sample(
        surrogate_model=surrogate_model,
        features=features,
        adj=adj,
        labels=labels,
        idx_test=idx_test,
        loss_fn=F.cross_entropy,
        perturbations=perturbations
    )

    ################################################
    # Train a model on the perturbed data
    ################################################

    print('==== Training model on attacked data ====')

    attacked_adj = get_modified_adj(adj, best)

    posisoned_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers
    ).to(device)

    posisoned_model.train()
    optimizer = torch.optim.Adam(
        posisoned_model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    for epoch in range(args.atk_epochs):

        predictions = train_step(
            model=posisoned_model,
            optimizer=optimizer,
            features=features,
            adj=attacked_adj,
            labels=labels,
            idx_train=idx_train,
            loss_fn=F.cross_entropy
        )

        show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

    ################################################
    # Evaluation
    ################################################
    model.eval()
    surrogate_model.eval()
    posisoned_model.eval()

    print("#" * 30)
    print(
        f"Evaluation ({args.ptb_rate:.2%} budget)")
    print("#" * 30)

    evaluate_acc(model, features, adj, labels,
                 idx_train, idx_test, "Baseline model on clean data:       ")

    evaluate_acc(posisoned_model, features, attacked_adj, labels,
                 idx_train, idx_test, "Poisoned model on perturbed data:   ")

    show_change_matrix(adj, attacked_adj, labels)

    # print("==== Model trained on clean graph ====")
    # evaluate_acc(model, features, adj, labels,
    #              idx_train, idx_test, "Clean:  \t")

    # evaluate_acc(model, features, attacked_adj,
    #              labels, idx_train, idx_test, "Perturbed:\t")

    # print("==== Surrogate model ====")
    # evaluate_acc(surrogate_model, features, adj, labels,
    #              idx_train, idx_test, "Clean:  \t")

    # evaluate_acc(surrogate_model, features, attacked_adj,
    #              labels, idx_train, idx_test, "Perturbed:\t")

    # print("==== Posioned model ====")
    # evaluate_acc(posisoned_model, features, adj, labels,
    #              idx_train, idx_test, "Clean:  \t")

    # evaluate_acc(posisoned_model, features, attacked_adj,
    #              labels, idx_train, idx_test, "Perturbed:\t")


if __name__ == "__main__":
    main()
    print("finished!")
