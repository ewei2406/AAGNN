import numpy as np
import torch.nn.functional as F
import torch


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


def random_sample(surrogate_model, features, adj, labels, idx_test, loss_fn, perturbations, k=10):
    min_loss = 1000
    with torch.no_grad():
        for i in range(k):
            sample = torch.bernoulli(perturbations)
            modified_adj = invert_by(adj, sample)

            sample_predictions = surrogate_model(
                features, modified_adj).squeeze()
            loss = loss_fn(
                sample_predictions[idx_test], labels[idx_test])
            if loss < min_loss:
                min_loss = loss
                best = sample

    print(f"Best sample: {int(best.sum())} edges \t Loss: {loss.item():.2f}")

    return best


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


def bisection(perturbations, a, b, n_perturbations, epsilon):

    def func(perturbations, x, n_perturbations):
        return torch.clamp(perturbations-x, 0, 1).sum() - n_perturbations
    
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