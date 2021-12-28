import torch
import torch.nn.functional as F

import os
import argparse
import numpy as np
from tqdm import tqdm

from aagnn.GCN import GCN
from aagnn.graphData import loadGraph
from aagnn import utils
from aagnn import metrics

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

    parser.add_argument('--protect_size', type=float, default=0.14,
                        help='Number of randomly chosen protected nodes')
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

    adj, labels, features, idx_train, idx_val, idx_test = loadGraph(args.data_dir, args.dataset, 'gcn', args.data_seed, device)
    
    ################################################
    # Baseline
    ################################################

    baseline = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name="baseline"
        )

    baseline.fit(features, adj, labels, idx_train, idx_test, args.reg_epochs)

    ################################################
    # Protected set
    ################################################

    g0 = torch.rand(features.shape[0]) <= args.protect_size
    print(f"Number of protected nodes: {g0.sum():.0f}")
    print(f"Protected Size: {g0.sum() / features.shape[0]:.2%}")
    
    # g0 = labels == args.protected_label
    g_g0 = ~g0

    ################################################
    # Perturbing
    ################################################

    surrogate = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name="surrogate"
        )
    
    surrogate.fit(features, adj, labels, idx_train, idx_test, args.surrogate_epochs)
    surrogate.eval()

    perturbations = torch.zeros_like(adj).float()
    perturbations.requires_grad = True

    num_perturbations = int(args.ptb_rate * (adj.sum() / 2))

    t = tqdm(range(args.ptb_epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    t.set_description("Perturbing")
    
    def loss_func(pred, labels):
        loss = 2 * F.cross_entropy(pred[g0], labels[g0]) \
            - F.cross_entropy(pred[g_g0], labels[g_g0]) \
            # - F.cross_entropy(pred, labels)
        
        return loss


    optimizer = torch.optim.Adam(
        surrogate.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    for epoch in t:
        surrogate.eval()
        modified_adj = utils.get_modified_adj(adj, perturbations)
        predictions = surrogate(features, modified_adj)

        loss = loss_func(predictions, labels)

        adj_grad = torch.autograd.grad(loss, perturbations)[0]

        lr = ((args.ptb_rate) * 5) / ((epoch+1) ** 2)
        
        perturbations = perturbations + (lr * adj_grad)

        perturbations = utils.projection(perturbations, num_perturbations)

        # Train surrogate
        if args.surrogate_train == "Y":
            modified_adj = utils.get_modified_adj(adj, perturbations)

            surrogate.train()
            optimizer.zero_grad()
            p = surrogate(features, modified_adj)

            surr_loss = F.cross_entropy(p[idx_train], labels[idx_train])
            surr_loss.backward()
            optimizer.step() 

            t.set_postfix({"adj loss": loss.item(),
                "adj_grad": int(adj_grad.sum()),
                "pre-projection": int(perturbations.sum()),
                "target": int(num_perturbations),
                "surrogate_loss": surr_loss.item()})

    with torch.no_grad():

        max_loss = -1000

        for k in range(0,3):
            sample = torch.bernoulli(perturbations)
            modified_adj = utils.get_modified_adj(adj, perturbations)

            predictions = surrogate(features, modified_adj)

            loss = loss_func(predictions, labels)

            if loss > max_loss:
                max_loss = loss
                best = sample
    
    print(f"Best sample loss: {loss:.2f}\t Edges: {best.sum() / 2:.0f}")

    ################################################
    # Train model on "locked" graph
    ################################################
    
    print("==== Training model on 'locked' data ====")

    locked_adj = utils.get_modified_adj(adj, best)

    locked_model = GCN(
        input_features=features.shape[1],
        output_classes=labels.max().item()+1,
        hidden_layers=args.hidden_layers,
        device=device,
        lr=args.model_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        name="locked_model"
        )

    locked_model.fit(features, locked_adj, labels, idx_train, idx_test, args.reg_epochs)

    ################################################
    # Evaluation
    ################################################
    
    baseline.eval()
    predictions = baseline(features, adj)

    locked_model.eval()
    lock_predictions = locked_model(features, locked_adj)

    b_g0 = metrics.acc(predictions[g0], labels[g0])
    b_g_g0 = metrics.acc(predictions[g_g0], labels[g_g0])
    b_g = metrics.acc(predictions, labels)

    l_g0 = metrics.acc(lock_predictions[g0], labels[g0])
    l_g_g0 = metrics.acc(lock_predictions[g_g0], labels[g_g0])
    l_g = metrics.acc(lock_predictions, labels)


    print("==== Accuracies ====")
    print(f"         Base\tLock\tChange")
    print(f"G0     | {b_g0:.2%}\t{l_g0:.2%}\t{l_g0-b_g0:.2%}")
    print(f"G - G0 | {b_g_g0:.2%}\t{l_g_g0:.2%}\t{l_g_g0-b_g_g0:.2%}")
    print(f"G      | {b_g:.2%}\t{l_g:.2%}\t{l_g-b_g:.2%}")

    # Metrics

    change = locked_adj - adj

    def count(bool_list):
        idx = utils.bool_to_idx(bool_list).squeeze()

        temp_adj = change.clone()
        temp_adj.index_fill_(dim=0, index=idx, value=0)
        diff = change - temp_adj

        temp_adj = diff.clone()
        temp_adj.index_fill_(dim=1, index=idx, value=0)
        diff = diff - temp_adj

        add = int(diff.clamp(0,1).sum() / 2)
        remove = int(diff.clamp(-1,0).abs().sum() / 2)

        return add, remove

    total_add = int(change.clamp(0,1).sum() / 2)
    total_remove = int(change.clamp(-1,0).abs().sum() / 2)

    g0g0_add, g0g0_remove = count(g0)
    gXgX_add, gXgX_remove = count(g_g0)

    g0gX_add = total_add - g0g0_add - gXgX_add
    g0gX_remove = total_remove - g0g0_remove - gXgX_remove

    print("==== Edges ====")
    print(f"          Add\tRemove")
    print(f"G0 - G0 | {g0g0_add}\t{g0g0_remove}")
    print(f"G0 - G  | {g0gX_add}\t{g0gX_remove}")
    print(f"G  - G  | {gXgX_add}\t{gXgX_remove}")
    print(f"Total   | {total_add}\t{total_remove}")
    print(f"Grand total: {total_add+total_remove}")

    ################################################
    # CSV
    ################################################
    
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
    print()
    main()
