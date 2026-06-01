from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import time

from graffe_core.models.residual_sage import ResidualSAGE
from graffe_core.models.base_sage import SAGE
from graffe_core.models.cagnn import CAGNN
from graffe_core.subgraph_sampler import SubgraphSampler
from graffe_core.explainability import FraudExplainer
from graffe_core.features.velocity import augment_with_velocity

# ── Configuration ──────────────────────────────────────────────
GRAPH_PATH = Path("data/elliptic_pyg_graph.pt")
RES_CKPT_PATH = Path("checkpoints/residualsage_best.pt")
BASE_CKPT_PATH = Path("checkpoints/sage_v2_best.pt")

print("Initializing Production Fraud Engine (CPU Mode)...")

# 1. Load Data
data = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
data = augment_with_velocity(data)

# 2. Load Models
# Base Model (165 features)
base_model = SAGE(in_channels=165, hidden_channels=64)
if BASE_CKPT_PATH.exists():
    try:
        base_model.load_state_dict(torch.load(BASE_CKPT_PATH, map_location="cpu", weights_only=False))
        print(f"Loaded Standard GraphSAGE weights from {BASE_CKPT_PATH}")
    except Exception as e:
        print(f"Base model weight loading error: {e}")
base_model.eval()

# Residual Model (170 features - includes velocity)
res_model = ResidualSAGE(in_channels=170, hidden_channels=32)
if RES_CKPT_PATH.exists():
    try:
        res_model.load_state_dict(torch.load(RES_CKPT_PATH, map_location="cpu", weights_only=False))
        print(f"Loaded ResidualSAGE weights from {RES_CKPT_PATH}")
    except Exception as e:
        print(f"Residual model weight loading error: {e}")
res_model.eval()

# CAGNN Model (170 features)
cagnn_model = CAGNN(in_channels=170, hidden_channels=64)
cagnn_model.eval()

# 3. Initialize Components
sampler = SubgraphSampler(data, max_neighbors=50)
explainer_res = FraudExplainer(res_model)
explainer_base = FraudExplainer(base_model)
explainer_cagnn = FraudExplainer(cagnn_model)

app = FastAPI(title="Transaction Intelligence Framework API")

class PredictionResponse(BaseModel):
    node_id: int
    fraud_score: float
    risk_level: str
    predicted_fraud: bool
    top_features: list
    influential_neighbors: list
    latency_ms: float
    burst_score: float
    fanout_ratio: float
    ttl_proxy: float
    model_used: str

FEATURE_NAMES = {
    165: "In-Degree Velocity",
    166: "Out-Degree Velocity",
    167: "Fan-out Ratio (Velocity)",
    168: "Burst Score (Velocity)",
    169: "Time-to-Live Proxy (Velocity)"
}

@app.get("/predict/{node_id}")
@app.post("/predict/{node_id}")
def predict(node_id: int, model: str = "Residual GraphSAGE (Production)"):
    start_time = time.perf_counter()
    
    if node_id < 0 or node_id >= data.num_nodes:
        raise HTTPException(status_code=400, detail="Invalid node_id")

    # Select model based on choice
    if "CAGNN" in model:
        current_model = cagnn_model
        current_explainer = explainer_cagnn
        use_velocity_in_model = True
    elif "Standard GraphSAGE" in model:
        current_model = base_model
        current_explainer = explainer_base
        use_velocity_in_model = False
    else:
        current_model = res_model
        current_explainer = explainer_res
        use_velocity_in_model = True

    # 1. Sample immediate neighborhood (1-hop)
    sub_data = sampler.get_subgraph(node_id, num_hops=1)
    
    # Restrict features if using base SAGE which expects 165 features
    x_input = sub_data.x if use_velocity_in_model else sub_data.x[:, :165]

    # 2. Run Inference
    with torch.no_grad():
        out = current_model(x_input, sub_data.edge_index)
        target_idx = sub_data.mapping
        score = torch.sigmoid(out[target_idx]).item()
    
    # 3. Explainability
    explanation = current_explainer.explain_node(target_idx, x_input, sub_data.edge_index)
    
    # Map features to human readable
    mapped_features = []
    for f_idx in explanation["top_features"]:
        if f_idx in FEATURE_NAMES:
            mapped_features.append(FEATURE_NAMES[f_idx])
        elif f_idx < 94:
            mapped_features.append(f"Local Transaction Feature #{f_idx}")
        else:
            mapped_features.append(f"Aggregated Neighborhood Feature #{f_idx}")
            
    latency = (time.perf_counter() - start_time) * 1000
    risk_level = "HIGH" if score > 0.7 else ("MEDIUM" if score > 0.4 else "LOW")
    
    # Extract structural velocity directly from pre-computed data
    # (Node's local velocity features are 167: fanout, 168: burst, 169: ttl)
    node_vfeats = data.x[node_id]
    
    return {
        "node_id": node_id,
        "fraud_score": round(score, 6),
        "risk_level": risk_level,
        "predicted_fraud": bool(score >= 0.5),
        "top_features": mapped_features,
        "influential_neighbors": explanation["influential_neighbors"],
        "latency_ms": round(latency, 2),
        "burst_score": round(node_vfeats[168].item(), 4),
        "fanout_ratio": round(node_vfeats[167].item(), 4),
        "ttl_proxy": round(node_vfeats[169].item(), 4),
        "model_used": "ResidualSAGE" if use_velocity_in_model else "Standard SAGE"
    }
