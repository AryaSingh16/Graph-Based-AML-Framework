import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ResidualSAGE(torch.nn.Module):
    """
    GraphSAGE with residual skip connections for improved gradient flow on CPU.
    Optimized for low-RAM (hidden_dim=32).
    """
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        
        # Projection for residual connection (if in != out)
        self.lin_skip = torch.nn.Linear(in_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        identity = self.lin_skip(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = x + identity  # Skip connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        identity = x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x + identity  # Skip connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output Layer
        x = self.conv3(x, edge_index)
        return x.view(-1)
