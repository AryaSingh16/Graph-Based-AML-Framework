import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EllipticGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)   # (N,)
