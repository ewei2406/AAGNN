import torch.nn.functional as F


def get_acc(prediction, truth, mask):
    """
    Cross entropy loss and accuracy
    """

    loss = F.cross_entropy(prediction[mask], target[mask]).item()
    acc = (prediction[mask] == target[mask]).sum() / mask.sum()

    return loss, acc


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

