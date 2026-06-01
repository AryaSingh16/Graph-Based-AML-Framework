# graffe_core/models/elliptic_gat.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EllipticGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, 1, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze()
