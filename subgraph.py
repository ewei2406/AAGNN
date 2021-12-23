import torch
import numpy as np
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

from deeprobust.graph.data import Dataset

import aagnn.utils as utils
import aagnn.metrics as metrics
import aagnn.dataLoading as dataLoading
from aagnn.training import train_step
from aagnn.models import GCN


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

    data = Dataset(
        root=args.data_dir, 
        name=args.dataset, 
        setting='gcn', 
        seed=args.data_seed)

    adj, features, labels, idx_train, idx_val, idx_test = \
        dataLoading.aagnn_format(data, device, args, verbose=True)

    print((labels == args.protected_label).sum())
    
    ################################################
    # Baseline
    ################################################
    
    print('==== Training regular model ====')

    baseline_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers
    ).to(device)

    baseline_model.train()

    optimizer = torch.optim.Adam(
        baseline_model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    t = tqdm(range(args.reg_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    t.set_description("Regular")

    for epoch in t:
        predictions = train_step(
            model=baseline_model,
            optimizer=optimizer,
            features=features,
            adj=adj,
            labels=labels,
            idx_train=idx_train,
            loss_fn=F.cross_entropy,
            iterator=t,
        )
    
    ################################################
    # Perturbing the data
    ################################################

    print('==== Perturbing ====')

    if args.edge_case == "add":
        target_idx = (labels == args.protected_label).int().nonzero().permute(1, 0).squeeze()

        # print(target_idx)

        perturbed = adj.clone()

        # print(perturbed.shape)

        perturbed.index_fill_(1, target_idx, 1)
        perturbed.index_fill_(0, target_idx, 1)

        best = perturbed

    elif args.edge_case == "remove":
        target_idx = (labels == args.protected_label).int().nonzero().permute(1, 0).squeeze()

        # print(target_idx)

        perturbed = adj.clone()

        # print(perturbed.shape)

        perturbed.index_fill_(1, target_idx, 0)
        perturbed.index_fill_(0, target_idx, 0)

        best = perturbed
    else:
        # Training surrogate
        surrogate = GCN(
            input_features=features.shape[1],
            output_classes=labels.max().item()+1,
            hidden_layers=args.hidden_layers
        ).to(device)

        surrogate.train()

        optimizer = torch.optim.Adam(
            surrogate.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

        t = tqdm(range(args.surrogate_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        t.set_description("Surrogate")
        for epoch in t:
            predictions = train_step(
                model=surrogate,
                optimizer=optimizer,
                features=features,
                adj=adj,
                labels=labels,
                idx_train=idx_train,
                loss_fn=F.cross_entropy,
                iterator=t,
            )
        
        # Perturbing
        surrogate.eval()

        perturbations = torch.zeros_like(adj).float()
        perturbations.requires_grad = True

        num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

        t = tqdm(range(args.ptb_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        t.set_description("Perturbing: ")

        g0 = [labels == args.protected_label]
        g_ex_g0 = [labels != args.protected_label]

        for epoch in t:
            # Perturb edges
            modified_adj = utils.invert_by(adj, perturbations)
            predictions = surrogate(features, modified_adj).squeeze()

            loss = F.cross_entropy(predictions[g0], labels[g0]) - F.cross_entropy(predictions[g_ex_g0], labels[g_ex_g0]) - F.cross_entropy(predictions, labels)

            adj_grad = torch.autograd.grad(loss, perturbations)[0]

            lr = ((args.ptb_rate) * 5) / ((epoch+1) ** 2)

            # diff = max(abs(perturbations.sum() - num_perturbations), num_perturbations)
            # mult = adj_grad * (diff / adj_grad.sum())

            perturbations = perturbations + (lr * adj_grad)

            # print(adj_grad.sum(), perturbations.sum(), num_perturbations)

            t.set_postfix({"adj_grad": int(adj_grad.sum())})
            t.set_postfix({"pre-projection": int(perturbations.sum())})
            t.set_postfix({"target": int(num_perturbations)})

            perturbations = utils.projection(perturbations, num_perturbations)

            t.set_postfix({"edges_perturbed": int(perturbations.sum())})

            # Train surrogate
            if args.surrogate_train == "Y":
                modified_adj = utils.invert_by(adj, perturbations)

                train_step(
                    model=surrogate,
                    optimizer=optimizer,
                    features=features,
                    adj=modified_adj,
                    labels=labels,
                    idx_train=idx_train,
                    loss_fn=F.cross_entropy,
                    iterator=t
                )

        perturbations = perturbations.clamp(0,1)
        best = utils.random_sample(
            surrogate_model=surrogate,
            features=features,
            adj=adj,
            labels=labels,
            idx_test=idx_test,
            loss_fn=lambda x, y: -F.cross_entropy(x, y),
            perturbations=perturbations
        )
    
    ################################################
    # Train model on "locked" graph
    ################################################
    
    print("==== Training model on 'locked' data ====")

    locked_adj = utils.invert_by(adj, best)

    locked_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers
    ).to(device)

    locked_model.train()

    optimizer = torch.optim.Adam(
        locked_model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    t = tqdm(range(args.reg_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for epoch in t:
        predictions = train_step(
            model=locked_model,
            optimizer=optimizer,
            features=features,
            adj=locked_adj,
            labels=labels,
            idx_train=idx_train,
            loss_fn=F.cross_entropy,
            iterator=t,
        )

    ################################################
    # Evaluate Accuracy
    ################################################

    
    all_labels = torch.unique(labels).numpy()
    excluding = all_labels[all_labels != args.protected_label]

    print("==== Baseline accuracies ====")

    baseline_model.eval()
    predictions = baseline_model(features, adj).squeeze()

    c_G = metrics.acc_by_label(predictions, labels, idx_test, all_labels)
    c_G0 = metrics.acc_by_label(predictions, labels, idx_test, [args.protected_label])
    c_G_G0 = metrics.acc_by_label(predictions, labels, idx_test, excluding)

    print(f"Base | G      : {c_G:.2%}")
    print(f"Base | G0     : {c_G0:.2%}")
    print(f"Base | G - G0 : {c_G_G0:.2%}")

    print("==== Locked accuracies ====")

    locked_model.eval()
    predictions = locked_model(features, locked_adj).squeeze()

    l_G = metrics.acc_by_label(predictions, labels, idx_test, all_labels)
    l_G0 = metrics.acc_by_label(predictions, labels, idx_test, [args.protected_label])
    l_G_G0 = metrics.acc_by_label(predictions, labels, idx_test, excluding)

    print(f"Lock | G      : {l_G:.2%}")
    print(f"Lock | G0     : {l_G0:.2%}")
    print(f"Lock | G - G0 : {l_G_G0:.2%}")

    print("==== Change ====")

    d_G = l_G - c_G
    d_G0 = l_G0 - c_G0
    d_G_G0 = l_G_G0 - c_G_G0

    print(f"Delta | G      : {d_G:.2%}")
    print(f"Delta | G0     : {d_G0:.2%}")
    print(f"Delta | G - G0 : {d_G_G0:.2%}")

    print("==== Edges ====")

    target_idx = labels == args.protected_label

    metrics.show_target_change_matrix(adj, locked_adj, labels, target_idx)

    
    if args.csv != '':
        csv_path = f"./{args.csv}.csv"

        file_exists = os.path.isfile(csv_path)

        with open (csv_path, 'a+') as f:
            
            if not file_exists:
                headers = [
                    'reg_epochs', 
                    'ptb_epochs', 
                    'surrogate_epochs', 
                    'ptb_rate','protected_label', 'retrain',
                    'c_G', 'c_G0', 'c_G_G0', 
                    'l_G', 'l_G0', 'l_G_G0', 
                    'd_G', 'd_G0', 'd_G_G0']
                f.write(",".join(headers) + "\n")

            data = [
                args.reg_epochs, 
                args.ptb_epochs,
                args.surrogate_epochs,
                args.ptb_rate, args.protected_label, args.surrogate_retrain,
                c_G, c_G0, c_G_G0,
                l_G, l_G0, l_G_G0,
                d_G, d_G0, d_G_G0
                ]
                
            f.write(",".join([str(x) for x in data]) + "\n")



if __name__ == "__main__":
    main()
