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
    parser.add_argument('--ptb_rate', type=float, default=0.5,
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

    # dataset = Planetoid(root='./tmp/Cora', name='Cora')
    # data = dataset[0].to(device)

    # features, adj, labels = data.x, to_adj(data.edge_index), data.y
    # idx_train, idx_test, idx_val = data.train_mask, data.test_mask, data.val_mask

    data = Dataset(root=args.data_dir, name=args.dataset,
                   setting='gcn', seed=args.data_seed)

    adj, features, labels = process(data, device)

    idx_train = indices_to_binary(data.idx_train, features.shape[0])
    idx_val = indices_to_binary(data.idx_val, features.shape[0])
    idx_test = indices_to_binary(data.idx_test, features.shape[0])

    print('==== Dataset ====')
    # print(f'density: {nx.density(nx.from_numpy_array(adj))}')
    print(f'adj shape: {list(adj.shape)}')
    print(f'feature shape: {list(features.shape)}')
    print(f'num labels: {labels.max().item()+1}')
    # print(f'split seed: {args.data_seed}')
    print(
        f'train|val|test: {idx_train.sum()}|{idx_val.sum()}|{idx_test.sum()}')

    ################################################
    # Train a model regularly 
    ################################################
    
    print('==== Training regular model ====')

    model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(10):

        predictions = train_step(model, optimizer, features, adj, labels, idx_train, F.cross_entropy)
        
        show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

        # optimizer.zero_grad()
        # predictions = model(features, adj).squeeze()

        # show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

        # loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
        # loss.backward()
        # optimizer.step()
    
    ################################################
    # Attack the data
    ################################################

    print('==== Attacking Data ====')

    num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

    surrogate_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1
    ).to(device)

    optimizer = torch.optim.Adam(
        surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)

    perturbations = torch.zeros_like(adj).float()

    perturbations.requires_grad = True

    for epoch in range(10):
        # Train the model =======================
        surrogate_model.train()
        modified_adj = get_modified_adj(adj, perturbations)
        # modified_adj = normalize_adj(modified_adj) # Normalize it????

        predictions = train_step(surrogate_model, optimizer, features,
                                 modified_adj, labels, idx_train, F.cross_entropy)
                                 
        show_acc(epoch+1, predictions, labels, idx_train, idx_test, False)

        # optimizer.zero_grad()
        # predictions = surrogate_model(features, modified_adj).squeeze()
        # show_acc(epoch+1, predictions, labels, idx_train, idx_test, False)

        # loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
        # loss.backward()
        # optimizer.step()

        # Update the adj matrix =================
        surrogate_model.eval()
        modified_adj = get_modified_adj(adj, perturbations)

        predictions = surrogate_model(features, modified_adj).squeeze()

        loss = -F.cross_entropy(predictions[idx_train], labels[idx_train])
        adj_grad = torch.autograd.grad(loss, perturbations)[0]

        # Update learning rate =================
        lr = 200 / (np.sqrt(epoch+1))
        perturbations = perturbations + (lr * adj_grad)
        perturbations = projection(perturbations, num_perturbations)

        print(f"Epoch: {epoch+1} \t Edges perturbed: {int(perturbations.sum())} \t Loss: {loss:.2f}")

    # Draw samples from best ==========
    min_loss = 1000
    k = 10
    with torch.no_grad():
        for i in range(k):
            sample = torch.bernoulli(perturbations)
            modified_adj = get_modified_adj(adj, sample)

            sample_predictions = surrogate_model(features, modified_adj).squeeze()
            loss = F.cross_entropy(sample_predictions[idx_test], labels[idx_test])
            if loss < min_loss:
                min_loss = loss
                best = sample
    
    print(f"Best sample: {int(best.sum())} edges \t Loss: {loss.item():.2f}")
    
    ################################################
    # Train a model on the perturbed data
    ################################################

    print('==== Training model on attacked data ====')

    attacked_adj = get_modified_adj(adj, best)

    posisoned_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1
    ).to(device)

    posisoned_model.train()
    optimizer = torch.optim.Adam(
        posisoned_model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(10):
        optimizer.zero_grad()
        predictions = posisoned_model(features, attacked_adj).squeeze()

        show_acc(epoch+1, predictions, labels, idx_train, idx_test, True)

        loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

    ################################################
    # Evaluation
    ################################################
    model.eval()
    surrogate_model.eval()
    posisoned_model.eval()

    print("==== Model trained on clean graph ====")
    print("- Performance on clean graph:")
    evaluate_acc(model, features, adj, labels, idx_train, idx_test)

    print("- Performance on perturbed graph:")
    evaluate_acc(model, features, attacked_adj,
                 labels, idx_train, idx_test)

    print("==== Surrogate model ====")
    print("- Performance on clean graph:")
    evaluate_acc(surrogate_model, features, adj, labels, idx_train, idx_test)

    print("- Performance on perturbed graph:")
    evaluate_acc(surrogate_model, features, attacked_adj,
                 labels, idx_train, idx_test)

    print("==== Posioned model ====")
    print("- Performance on clean graph:")
    evaluate_acc(posisoned_model, features, adj, labels, idx_train, idx_test)

    print("- Performance on perturbed graph:")
    evaluate_acc(posisoned_model, features, attacked_adj,
                 labels, idx_train, idx_test)











if __name__ == "__main__":
    main()
    print("finished!")
