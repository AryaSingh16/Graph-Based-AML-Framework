import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TemporalSAGE(torch.nn.Module):
    """
    GraphSAGE variant that incorporates temporal snapshots (timesteps) into the logic.
    Optimized for CPU inference.
    """
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, dropout=0.3):
        super().__init__()
        # Time embedding layer: 50 timesteps -> 8-dim embedding
        self.time_emb = torch.nn.Embedding(51, 8) 
        
        self.conv1 = SAGEConv(in_channels + 8, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, steps=None):
        """
        x: Node features
        steps: Timestep for each node
        """
        if steps is not None:
            t_emb = self.time_emb(steps)
            x = torch.cat([x, t_emb], dim=1)
        else:
            # Fallback if no steps provided
            t_emb = torch.zeros((x.size(0), 8), device=x.device)
            x = torch.cat([x, t_emb], dim=1)
            
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)
