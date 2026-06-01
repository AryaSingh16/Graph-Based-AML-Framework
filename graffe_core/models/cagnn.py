import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree


class CAGNN(torch.nn.Module):
    """
    CAGNN: Camouflage-Aware Graph Neural Network
    =============================================
    Designed specifically for fraud detection on cryptocurrency transaction graphs.

    Core Insight:
        Standard GNNs assume homophily — fraudulent nodes cluster together.
        In real AML scenarios (e.g., Elliptic Bitcoin dataset), fraudsters
        deliberately route transactions through LEGITIMATE wallets to camouflage
        their trail. This is called graph camouflage or heterophily.

        Standard GNNs smooth out this suspicious signal when averaging neighbors,
        effectively hiding the fraud. CAGNN solves this with two parallel branches:

        1. Low-pass branch  — standard neighbor aggregation (homophily signal)
        2. High-pass branch — amplifies DIFFERENCES from neighbors (heterophily/camouflage signal)
        3. Adaptive gate α  — learned per-node weight between the two branches

    Architecture:
        Layer 1: Dual-branch aggregation (low-pass + high-pass) with gate α
        Layer 2: Second dual-branch aggregation
        Output:  Linear(hidden → 1) → fraud logit

    Parameters:
        in_channels     : Number of input features per node (165 or 170 with velocity)
        hidden_channels : Hidden representation size (default 64)
        dropout         : Dropout probability (default 0.3)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()

        # --- Low-pass branch (standard homophily aggregation) ---
        self.low_conv1 = SAGEConv(in_channels, hidden_channels)
        self.low_conv2 = SAGEConv(hidden_channels, hidden_channels)

        # --- High-pass branch (heterophily / camouflage detection) ---
        # Uses a linear layer on (node_feat - mean_neighbor_feat) to amplify differences
        self.high_lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.high_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        # --- Adaptive gate (learnable per-layer scalar, shared across nodes) ---
        # Initialized at 0.5 — learns to emphasize camouflage signal for fraud nodes
        self.gate1 = torch.nn.Parameter(torch.tensor(0.5))
        self.gate2 = torch.nn.Parameter(torch.tensor(0.5))

        # --- Output ---
        self.lin_out = torch.nn.Linear(hidden_channels, 1)

        self.dropout = dropout

    def _high_pass(self, x: torch.Tensor, edge_index: torch.Tensor, lin: torch.nn.Linear) -> torch.Tensor:
        """
        High-pass filter: computes (node_feat - mean_neighbor_feat).
        A node that looks very different from its neighbors gets a high signal here.
        This is exactly what happens when a fraudster hides among legitimate nodes.
        """
        row, col = edge_index
        # Compute mean of neighbor features for each node
        num_nodes = x.size(0)
        neighbor_sum = torch.zeros(num_nodes, x.size(1), device=x.device)
        neighbor_sum.scatter_add_(0, row.unsqueeze(1).expand_as(x[col]), x[col])
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1).unsqueeze(1)
        neighbor_mean = neighbor_sum / deg

        # High-pass = difference from neighborhood mean
        diff = x - neighbor_mean
        return F.relu(lin(diff))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # ── Layer 1 ──────────────────────────────────────────
        # Low-pass: standard aggregation
        x_low = F.relu(self.low_conv1(x, edge_index))

        # High-pass: camouflage detection (amplify differences from neighbors)
        x_high = self._high_pass(x, edge_index, self.high_lin1)

        # Gate blend: α * low + (1-α) * high
        alpha1 = torch.sigmoid(self.gate1)
        x = alpha1 * x_low + (1.0 - alpha1) * x_high
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Layer 2 ──────────────────────────────────────────
        x_low = F.relu(self.low_conv2(x, edge_index))
        x_high = self._high_pass(x, edge_index, self.high_lin2)

        alpha2 = torch.sigmoid(self.gate2)
        x = alpha2 * x_low + (1.0 - alpha2) * x_high
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Output ───────────────────────────────────────────
        return self.lin_out(x).squeeze(-1)
