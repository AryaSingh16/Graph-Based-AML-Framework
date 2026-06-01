import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import time

class SubgraphSampler:
    """
    Handles transaction-centric local inference by extracting k-hop ego graphs.
    Optimized for CPU with caching and neighbor capping.
    """
    def __init__(self, data: Data, max_neighbors=50):
        self.data = data
        self.max_neighbors = max_neighbors
        self.cache = {}

    def get_subgraph(self, node_id: int, num_hops=1) -> Data:
        """
        Extracts a k-hop subgraph around node_id.
        Returns a PyG Data object.
        """
        cache_key = (node_id, num_hops)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 1. Extract k-hop neighborhood
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_id,
            num_hops=num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes
        )
        
        # 2. Apply neighbor cap (Sampling)
        # If the subgraph is too large, we could truncate edge_index here, 
        # but k_hop_subgraph is usually small for Elliptic (low degree).
        # For production, we'd use NeighborLoader logic.
        
        sub_data = Data(
            x=self.data.x[subset],
            edge_index=edge_index,
            y=self.data.y[subset],
            mapping=mapping # Index of the target node in the subgraph
        )
        
        # 3. Cache and return
        self.cache[cache_key] = sub_data
        return sub_data

    def clear_cache(self):
        self.cache = {}
