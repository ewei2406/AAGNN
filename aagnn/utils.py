import numpy as np
import torch.nn.functional as F
import torch


def idx_to_bool(idx, max_len=None):
    """
    Converts an array of indices into a boolean array (where desired indices are True)
    
    Parameters
    ---
    idx : array-like
        array of indices
    max_len : int (optional)
        length of output
    
    Returns
    ---
    out : array
        array of booleans
    
    Examples
    ---
    >>>example
    
    """
    
    if not max_len:
        max_len = max(idx)
    
    arr = torch.zeros(max_len)
    arr[idx] = 1
    return arr > 0


def bool_to_idx(bool_list):
    """
    Converts a boolean array (where desired indices are True) into an array of indices
    
    Parameters
    ---
    bool_list : array-like
        array of booleans
    
    Returns
    ---
    out : array
        array of indices
    
    Examples
    ---
    >>>example
    
    """
    return bool_list.nonzero()


def to_adj(edge_ind):
    """
    Converts a list of edges to an adjacency matrix.
    """
    return torch.sparse_coo_tensor(edge_ind, torch.ones_like(edge_ind[0])).to_dense()


def to_edges(adj):
    """
    Converts an adjacency matrix to a list of edges
    """
    res = adj.float().nonzero().permute(1, 0)
    return res


def make_symmetric(adj):
    """
    Makes adj. matrix symmetric about the diagonal and sets the diagonal to 0.
    Keeps the upper triangle.
    """
    upper = torch.triu(adj)

    lower = torch.rot90(torch.flip(
        torch.triu(adj, diagonal=1), [0]), 3, [0, 1])

    result = (upper + lower).fill_diagonal_(0)
    return result


def get_modified_adj(adj, perturbations):
    """
    Inverts the adjacency matrix by a perturbation matrix (where 1 is to perturb, 0 is to not perturb)
    Uses only the bottom triangle of the perturbation matrix
    """

    tri = (adj + perturbations) - torch.mul(adj * perturbations, 2)
    return tri


def projection(perturbations, n_perturbations):
    """
    Get the projection of a perturbation matrix such that the sum over the distribution of perturbations is n_perturbations 
    
    Parameters
    ---
    perturbations : torch.tensor
        probability distribution of perturbations
    n_perturbations : int
        desired number of perturbations
    
    Returns
    ---
    out : torch.tensor
        projected perturbation matrix
    
    Examples
    ---
    >>>example
    
    """

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



# DO NOT USE!!!
# def random_sample(surrogate_model, features, adj, labels, idx_test, loss_fn, perturbations, k=10):
#     min_loss = 1000
#     with torch.no_grad():
#         for i in range(k):
#             sample = torch.bernoulli(perturbations)
#             modified_adj = invert_by(adj, sample)

#             sample_predictions = surrogate_model(
#                 features, modified_adj).squeeze()
#             loss = loss_fn(
#                 sample_predictions[idx_test], labels[idx_test])
#             if loss < min_loss:
#                 min_loss = loss
#                 best = sample

#     print(f"Best sample: {int(best.sum())} edges \t Loss: {loss.item():.2f}")

#     return best
