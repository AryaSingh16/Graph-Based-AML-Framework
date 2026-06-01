# Graph-Based AML framework 🛡️

An AML transaction intelligence framework for cryptocurrency fraud detection that evaluates classical machine learning, graph neural networks, and heterophily-aware graph architectures on the **Elliptic Bitcoin dataset**.

## The Research Arc
Many graph neural network architectures perform best under homophily, where connected nodes tend to share similar labels. However, cryptocurrency laundering networks frequently exhibit **heterophilic behavior**, where illicit transactions are intentionally embedded within predominantly legitimate neighborhoods.

Through rigorous ablation studies on the Elliptic dataset, this project reveals the opposite: **Graph Camouflage**. Fraudsters deliberately route illicit funds through highly legitimate nodes to blend their topological signatures. When standard GNNs average neighborhood features, they actively smooth away the fraud signal, leading to poorer performance than classical feature-only models.

To solve this, we introduced **CAGNN (Camouflage-Aware GNN)**, a custom architecture with a dual-branch (low-pass + high-pass) aggregation mechanism that explicitly detects heterophilic differences, and deployed a hybrid architecture:
1. **Scoring Engine**: Random Forest (Highest pure accuracy, fastest inference)
2. **Explainability Engine**: GNN Subgraph Sampling (Traces suspicious neighborhood flows for analysts)

## Models Evaluated

| Paradigm | Architecture | Description |
|---|---|---|
| **Novel / Custom** | **CAGNN** | Camouflage-Aware GNN with dual-branch homophily/heterophily gates |
| **Classical ML** | Random Forest | Baseline feature-only model |
| **Classical ML** | XGBoost | Boosted tree baseline |
| **Spatial GNN** | ResidualSAGE | GraphSAGE augmented with residual skip connections |
| **Spatial-Temporal GNN** | TemporalSAGE | Time-step embedded SAGE for concept drift |
| **Spatial GNN** | GraphSAGE | Standard inductive scalable graph network |
| **Spatial GNN** | EllipticGAT | Attention-weighted graph network |
| **Spatial GNN** | EllipticGCN | Standard convolutional graph network |

## Key Discoveries
See `FINDINGS.md` for a detailed breakdown of the research discoveries:
- **Graph Structure as a Weak Predictive Signal:** In our experiments, Random Forest consistently outperformed standard message-passing GNNs on the Elliptic dataset.
- **Velocity Features:** Velocity-derived features contributed limited predictive value relative to transaction-level financial features.
- **CAGNN Effectiveness:** By detecting heterophilic anomalies rather than smoothing them, CAGNN recovers much of the performance lost by standard spatial GNNs.

## Quick Start
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run the benchmark:** `python benchmark_runner.py`
3. **Launch the API:** `uvicorn scripts.main:app --reload`
4. **Launch the Dashboard:** `streamlit run dashboard/app.py`

## Repository Structure

```text
graph-neural-fraud-engine/
│
├── README.md                 # Project overview and instructions
├── FINDINGS.md               # Detailed research discoveries and ablation results
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes large datasets and checkpoints
├── Dockerfile                # Containerization setup
│
├── benchmark_runner.py       # Main script to train and evaluate all 8 models
├── latency_profiler.py       # Profiles inference speeds 
│
├── graffe_core/              # Core Machine Learning Library
│   ├── models/               
│   │   ├── cagnn.py          # Camouflage-Aware GNN (Novel)
│   │   ├── base_sage.py      # Standard GraphSAGE
│   │   ├── elliptic_gat.py   # GAT implementation
│   │   ├── elliptic_gcn.py   # GCN implementation
│   │   ├── residual_sage.py  # Residual GraphSAGE
│   │   └── temporal_sage.py  # Time-aware GraphSAGE
│   ├── features/
│   │   └── velocity.py       # Topological feature engineering logic
│   ├── explainability.py     # GNNExplainer logic for analyst reasoning
│   └── subgraph_sampler.py   # 1-hop sampler for ultra-fast local inference
│
├── scripts/                  
│   └── main.py               # FastAPI backend for model serving
│
└── dashboard/                
    └── app.py                # 4-Tab Streamlit Interface
```

