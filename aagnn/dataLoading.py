import numpy as np
import torch.nn.functional as F
import torch


def csr_to_tensor(csr):
    """
    Returns torch.tensor of a scipy.sparse.csr_matrix

    Parameters
    ---
    csr : scipy.sparse_csr_matrix

    Returns
    ---
    out : equivalent torch.tensor

    Examples
    ---

    """

    numpy_array = csr.toarray()
    tensor = torch.from_numpy(numpy_array)
    return tensor.long()


def indices_to_binary(indices, length):
    """
    Converts an tensor of indices to a tensor of booleans with true at corresponding indices
    
    Parameters
    ---
    indices : torch.tensor
        A tensor of indices
    length : int
        Length of new tensor (must be greater than maximum index in indices)
    
    Returns
    ---
    out : torch.tensor
        A tensor of booleans with length 'length'
    
    Examples
    ---
    
    """

    arr = torch.zeros(length)
    arr[indices] = 1
    return arr > 0


def process(data, device):
    """
    Converts deeprobust data to torch.tensor
    
    Parameters
    ---
    data : deeprobust.graph.data.Dataset
        Input data
    
    Returns
    ---
    out : torch.tensor, torch.tensor, torch.tensor
        boolean adjacency matrix, node features, node labels
    
    Examples
    ---
    
    """
    
    labels = torch.LongTensor(data.labels)
    features = torch.FloatTensor(np.array(data.features.todense()))
    adj = torch.LongTensor(data.adj.todense())

    return adj, features, labels


def aagnn_format(data, device, args, verbose=True):
    """
    Converts deeprobust data to GCN usable torch.tensor
    
    Parameters
    ---
    data : deeprobust.graph.data.Dataset
        Input data
    device : str
        Torch device ("gpu" or "cpu")
    args : args
        Arguments used by argparse
    verbose : bool
        Enable displaying of dataset summary
    
    Returns
    ---
    out : torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor

        adj, features, labels, idx_train, idx_val, idx_test

        Adjacency matrix, node features, node labels,
        Training set, Validation set, Testing set
    
    Examples
    ---
    Load a deeprobust dataset and transform to torch.tensor

    >>> data = deeprobust.graph.data.Dataset(
            root=args.data_dir, 
            name=args.dataset, 
            setting='gcn', 
            seed=args.data_seed)

    >>> adj, features, labels, idx_train, idx_val, idx_test = dataLoading.aagnn_format(data, device, args, verbose=True)
    
    """
    
    adj, features, labels = process(data, device)

    idx_train = indices_to_binary(data.idx_train, features.shape[0])
    idx_val = indices_to_binary(data.idx_val, features.shape[0])
    idx_test = indices_to_binary(data.idx_test, features.shape[0])

    if verbose:
        print('==== Dataset ====')
        print(f'adj shape: {list(adj.shape)}')
        print(f'feature shape: {list(features.shape)}')
        print(f'num labels: {labels.max().item()+1}')
        print(f'split seed: {args.data_seed}')
        print(
            f'train|val|test: {idx_train.sum()}|{idx_val.sum()}|{idx_test.sum()}')
    
    return adj, features, labels, idx_train, idx_val, idx_test
