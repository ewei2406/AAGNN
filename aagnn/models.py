import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.nn import DenseGCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_features, output_classes, hidden_layers=64):
        super().__init__()
        self.conv1 = DenseGCNConv(input_features, hidden_layers)
        self.conv2 = DenseGCNConv(hidden_layers, output_classes)

    def forward(self, x, adj):
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)

        return F.log_softmax(x, dim=1)
