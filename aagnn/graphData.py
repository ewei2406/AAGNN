import torch
import numpy as np

from aagnn.dataset import Dataset
import aagnn.utils as utils


def loadGraph(root, name, setting, seed, device, verbose=True):
    data = Dataset(root, name, setting, seed)

    adj = torch.LongTensor(data.adj.todense()).to(device)
    adj = utils.make_symmetric(adj)
    labels = torch.LongTensor(data.labels).to(device)
    features = torch.FloatTensor(np.array(data.features.todense())).to(device)

    def indices_to_bool(indices, length):
        arr = torch.zeros(length)
        arr[indices] = 1
        return arr > 0

    idx_train = utils.idx_to_bool(data.idx_train, features.shape[0])
    idx_val = utils.idx_to_bool(data.idx_val, features.shape[0])
    idx_test = utils.idx_to_bool(data.idx_test, features.shape[0])

    if verbose:
        print()
        print(f'==== Dataset Summary: {name} ====')
        print(f'adj shape: {list(adj.shape)}')
        print(f'feature shape: {list(features.shape)}')
        print(f'num labels: {labels.max().item()+1}')
        print(f'split seed: {seed}')
        print(
            f'train|val|test: {idx_train.sum()}|{idx_val.sum()}|{idx_test.sum()}')
    
    return adj, labels, features, idx_train, idx_val, idx_test


def loadData(root, seed, device, verbose=True):
    edges = np.loadtxt(root, delimiter="\t", dtype=int)
    edges = torch.from_numpy(edges).t()
    adj = utils.to_adj(edges)

    return adj


class Graph:
    def __init__(self, root, name, setting, seed, device):
        data = Dataset(root, name, setting, seed)

        self.seed = seed

        self.adj = torch.LongTensor(data.adj.todense()).to(device)
        self.labels = torch.LongTensor(data.labels).to(device)
        self.features = torch.FloatTensor(np.array(data.features.todense())).to(device)

        def indices_to_bool(indices, length):
            arr = torch.zeros(length)
            arr[indices] = 1
            return arr > 0

        self.idx_train = utils.idx_to_bool(data.idx_train, self.features.shape[0])
        self.idx_val = utils.idx_to_bool(data.idx_val, self.features.shape[0])
        self.idx_test = utils.idx_to_bool(data.idx_test, self.features.shape[0])

        self.name = name
    
    def summarize(self):
        print()
        print(f'==== Dataset Summary: {self.name} ====')
        print(f'adj shape: {list(self.adj.shape)}')
        print(f'feature shape: {list(self.features.shape)}')
        print(f'num labels: {self.labels.max().item()+1}')
        print(f'split seed: {self.seed}')
        print(
            f'train|val|test: {self.idx_train.sum()}|{self.idx_val.sum()}|{self.idx_test.sum()}')

    def make_symmetric_(self):
        self.adj = utils.make_symmetric(self.adj)

    def modify_adj_(self, perturbations):
        self.adj = utils.get_modified_adj(self.adj, perturbations)
    
    def get_edges(self):
        return utils.to_edges(self.adj)
    