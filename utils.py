from deeprobust.graph.global_attack import BaseAttack
from deeprobust.graph import utils
from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import optim
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import torch


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


def evaluate_acc(model, features, adj, labels, idx_train, idx_test, prefix=""):
    predictions = model(features, adj).squeeze()
    show_acc(False, predictions, labels, idx_train, idx_test, True, prefix)


def to_adj(edge_ind):
    """
    Converts a list of edges to an adjacency matrix.
    """
    return torch.sparse_coo_tensor(edge_ind, torch.ones_like(edge_ind[0])).to_dense()


def to_index(adj):
    """
    Converts an adjacency matrix to a list of edges
    """
    res = adj.float().nonzero().permute(1, 0)
    return res


def mirror(adj):
    """
    Returns the upper triangle mirrored onto the lower triangle.
    """
    upper = torch.triu(adj)
    lower = torch.rot90(torch.flip(
        torch.triu(adj, diagonal=1), [0]), 3, [0, 1])
    return upper + lower


def set_diagonal(adj, target=0):
    """
    Returns a copy of the matrix with the diagonal filled.
    """
    copy = adj.clone()
    return copy.fill_diagonal_(target)


def make_symmetric(adj):
    """
    Makes adj. matrix symmetric about the diagonal and sets the diagonal to 0.
    Keeps the upper triangle.
    """
    upper = torch.triu(adj)

    lower = torch.rot90(torch.flip(
        torch.triu(adj, diagonal=1), [0]), 3, [0, 1])

    result = set_diagonal(upper + lower, 0)
    return result


def invert_by(adj, adj_changes):
    """
    Inverts the adjacency matrix by a perturbation matrix (where 1 is to perturb, 0 is to not perturb)
    """
    return (adj + adj_changes) - torch.mul(adj * adj_changes, 2)


def normalize_adj(adj):
    """
    Normalizes adjacency matrix
    """
    device = adj.device
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def get_modified_adj(adj, perturbations):
    # adj_changes_square = perturbations - \
    #     torch.diag(torch.diag(perturbations, 0))
    # # ind = np.diag_indices(adj_changes.shape[0]) # this line seems useless
    # adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)

    # adj_changes_square = torch.clamp(adj_changes_square, -1, 1)

    # modified_adj = adj_changes_square + adj

    modified_adj = invert_by(adj, perturbations)
    
    return modified_adj


def projection(perturbations, n_perturbations):
    # projected = torch.clamp(self.adj_changes, 0, 1)
    if torch.clamp(perturbations, 0, 1).sum() > n_perturbations:
        left = (perturbations - 1).min()
        right = perturbations.max()
        miu = bisection(perturbations, left, right, n_perturbations, epsilon=1e-5)
        perturbations.data.copy_(torch.clamp(
            perturbations.data - miu, min=0, max=1))
    else:
        perturbations.data.copy_(torch.clamp(
            perturbations.data, min=0, max=1))
    
    return perturbations


def func(perturbations, x, n_perturbations):
    return torch.clamp(perturbations-x, 0, 1).sum() - n_perturbations


def bisection(perturbations, a, b, n_perturbations, epsilon):
    miu = a
    while ((b-a) >= epsilon):
        miu = (a+b)/2
        # Check if middle point is root
        if (func(perturbations, miu, n_perturbations) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(perturbations, miu, n_perturbations)*func(perturbations, a, n_perturbations) < 0):
            b = miu
        else:
            a = miu
    # print("The value of root is : ","%.4f" % miu)
    return miu

def csr_to_tensor(csr):
    numpy_array = csr.toarray()
    tensor = torch.from_numpy(numpy_array)
    return tensor.long()


def indices_to_binary(indices, length):
    arr = torch.zeros(length)
    arr[indices] = 1
    return arr > 0


def process(data, device):
    labels = torch.LongTensor(data.labels)
    features = torch.FloatTensor(np.array(data.features.todense()))
    adj = torch.LongTensor(data.adj.todense())

    return adj, features, labels


def edge_types(adj, labels):
    edges = to_index(adj).t().squeeze()
    similarities = []
    for edge in edges:
        is_same = labels[edge[0]] == labels[edge[1]]
        similarities.append(is_same)
    
    similarities = np.array(similarities)

    return int(similarities.sum() / 2), int((len(similarities) - similarities.sum()) / 2)


def show_change_matrix(adj, perturbed, labels):
    same, diff = edge_types(adj, labels)
    
    print(
        f"Clean || Same: {same} \t Different: {diff}")

    p_same, p_diff = edge_types(perturbed, labels)

    print(
        f"Atk   || Same: {p_same} \t Different: {p_diff}")

    print(f"Diff  || Same: {p_same - same} \t Different: {p_diff - diff}")
