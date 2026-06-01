import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree

class VelocityFeatureBuilder:
    """
    Computes temporal velocity and topological flow features for the Elliptic dataset.
    Optimized for CPU using vectorized operations.
    """
    def __init__(self, data: Data):
        self.data = data
        self.num_nodes = data.x.size(0)
        
    def build_all(self) -> torch.Tensor:
        """Computes all velocity features and returns a tensor (N, K)."""
        print("Computing Velocity Features (CPU Optimized)...")
        
        # 1. Degrees (In/Out Flow)
        row, col = self.data.edge_index
        out_degree = degree(row, self.num_nodes)
        in_degree = degree(col, self.num_nodes)
        
        # 2. Fan-out Ratio
        fanout_ratio = out_degree / (in_degree + 1e-6)
        
        # 3. Timestep counts (Burst Score proxy)
        # Count nodes per timestep
        step_counts = torch.bincount(self.data.step.long())
        node_step_counts = step_counts[self.data.step.long()].float()
        burst_score = node_step_counts / (step_counts.float().mean() + 1e-6)
        
        # 4. Hop/Flow indicators
        # We can use some of the existing 166 features if needed, 
        # but let's derive new ones from the graph structure.
        
        # Placeholder for 'ttl' (Time to Live) - heuristic: 
        # If a node has high out-degree but was created in a busy timestep.
        ttl_proxy = 1.0 / (out_degree + 1.0)
        
        # Stack features
        velocity_feats = torch.stack([
            in_degree,
            out_degree,
            fanout_ratio,
            burst_score,
            ttl_proxy
        ], dim=1)
        
        # Add basic volume features if available (First column of x is often a proxy for volume)
        # In Elliptic, columns 2-93 are local features.
        # Let's add a few more structural ratios.
        
        print(f"Generated {velocity_feats.size(1)} velocity features.")
        return velocity_feats

def augment_with_velocity(data: Data) -> Data:
    builder = VelocityFeatureBuilder(data)
    v_feats = builder.build_all()
    data.x = torch.cat([data.x, v_feats], dim=1)
    return data
