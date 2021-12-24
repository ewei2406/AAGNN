import torch
import numpy as np

from aagnn.dataset import Dataset
import aagnn.utils as utils

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
    