import numpy as np
import torch.nn.functional as F
import torch


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


def aagnn_format(data, device, args, verbose=True):
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