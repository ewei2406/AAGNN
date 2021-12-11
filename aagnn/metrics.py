import numpy as np
import torch.nn.functional as F
import torch
from .utils import to_index


def show_acc(epoch, predictions, labels, idx_train, idx_test, verbose=False, prefix=""):
    train_correct = (predictions.argmax(
        1)[idx_train] == labels[idx_train]).sum()
    train_acc = (train_correct) / (idx_train.sum())

    test_correct = (predictions.argmax(1)[idx_test] == labels[idx_test]).sum()
    test_acc = (test_correct) / (idx_test.sum())

    loss = F.cross_entropy(predictions[idx_train], labels[idx_train])

    if verbose:
        if epoch:
            print(
                f"{prefix}Epoch: {epoch} \t Train: {train_acc:.2%} \t Test: {test_acc:.2%} \t Loss: {loss:.2f}")
        else:
            print(
                f"{prefix}Train: {train_acc:.2%} \t Test: {test_acc:.2%} \t Loss: {loss:.2f}")
    
    return train_acc, test_acc


def evaluate_acc(model, features, adj, labels, idx_train, idx_test, prefix=""):
    predictions = model(features, adj).squeeze()
    return show_acc(False, predictions, labels, idx_train, idx_test, True, prefix)


def edge_types(adj, labels):
    edges = to_index(adj).t().squeeze()
    similarities = []
    for edge in edges:
        is_same = labels[edge[0]] == labels[edge[1]]
        similarities.append(is_same)
    
    similarities = np.array(similarities)

    return int(similarities.sum()), int((len(similarities) - similarities.sum()))


def summarize_edges (adj, perturbed, labels):
    same, diff = edge_types(adj, labels)
    p_same, p_diff = edge_types(perturbed, labels)

    print(f"")
    print(f"Edge Summary")
    print(f"        Same\tDiff")
    print(f"      +------------------")
    print(f"Clean | {same}\t{diff}")
    print(f"Atk   | {p_same}\t{p_diff}")
    print(f"Diff  | {p_same - same}\t{p_diff - diff}")
    print(f"      +------------------")


def same_diff(edges, labels):
    similarities = []
    for edge in edges:
        is_same = labels[edge[0]] == labels[edge[1]]
        similarities.append(is_same)

    similarities = np.array(similarities)
    return int(similarities.sum()), int((len(similarities) - similarities.sum()))


def show_change_matrix(adj, perturbed, labels):
    diff = perturbed - adj
    added = to_index(diff.clamp(0, 1)).t()
    removed = to_index(diff.clamp(-1, 0).abs()).t()

    s_added, d_added = same_diff(added, labels)
    s_removed, d_removed = same_diff(removed, labels)

    total = s_added + d_added + s_removed + d_removed

    print(f"")
    print(f"Changes Applied")
    print(f"          Same\tDiff")
    print(f"        +---------------")
    print(f"Added   | {s_added}\t{d_added}")
    print(f"Removed | {s_removed}\t{d_removed}")
    print(f"        +---------------")
    print(f"Total Chages={total}")
