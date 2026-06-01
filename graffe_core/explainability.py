import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data

class FraudExplainer:
    """
    Provides reasoning for fraud predictions using GNNExplainer.
    Optimized for CPU by using a limited number of iterations.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=100), # Reduced epochs for CPU
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='raw',
            ),
        )

    def explain_node(self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """
        Explains the prediction for a specific node.
        Returns top features and influential neighbors.
        """
        explanation = self.explainer(x, edge_index, index=node_idx)
        
        # 1. Top Features
        feat_importance = explanation.node_mask.sum(dim=0)
        top_feat_indices = torch.topk(feat_importance, k=5).indices.tolist()
        
        # 2. Influential Edges / Neighbors
        # We can look at the edge_mask to see which connections were most important
        edge_influence = explanation.edge_mask
        top_edge_indices = torch.topk(edge_influence, k=min(5, edge_influence.size(0))).indices.tolist()
        
        # Map back to neighbor IDs
        influential_neighbors = []
        for idx in top_edge_indices:
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            neighbor = src if dst == node_idx else dst
            influential_neighbors.append(int(neighbor))
            
        return {
            "top_features": top_feat_indices,
            "influential_neighbors": influential_neighbors,
            "explanation_obj": explanation
        }
