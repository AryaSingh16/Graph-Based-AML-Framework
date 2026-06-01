"""
Final Benchmark Runner — Graph Neural Fraud Engine
===================================================
Evaluates all 6 GNN models + 2 classical baselines on the Elliptic Bitcoin Dataset.
Results saved to results/final_benchmark_all_models.csv
"""

import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from graffe_core.models.elliptic_gcn import EllipticGCN
from graffe_core.models.elliptic_gat import EllipticGAT
from graffe_core.models.base_sage import SAGE
from graffe_core.models.residual_sage import ResidualSAGE
from graffe_core.models.temporal_sage import TemporalSAGE
from graffe_core.models.cagnn import CAGNN


DEVICE = torch.device("cpu")
SEEDS  = 3
EPOCHS = 150
POS_WEIGHT = 50.0   # Handle heavy class imbalance (1 illicit : ~50 licit)

Path("results").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    data = torch.load("data/elliptic_pyg_graph.pt", map_location=DEVICE, weights_only=False)
    mask       = data.y != -1
    train_mask = mask & (data.step < 35)
    test_mask  = mask & (data.step >= 35)
    return data, train_mask, test_mask


# ─────────────────────────────────────────────────────────────────────────────
# GNN Training + Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_gnn(model_class, model_kwargs, data, train_mask, test_mask, label):
    print(f"\n[GNN] {label} ...")
    pw = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    y = data.y.float()
    seed_results = []

    for seed in range(SEEDS):
        torch.manual_seed(seed)
        model = model_class(**model_kwargs).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        t0 = time.time()
        model.train()
        for epoch in range(EPOCHS):
            opt.zero_grad()
            out  = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            opt.step()

        train_time = time.time() - t0
        epoch_time = train_time / EPOCHS

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()

        y_true = y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        y_prob = probs[test_mask].cpu().numpy()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        seed_results.append({
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "precision":  precision_score(y_true, y_pred, zero_division=0),
            "recall":     recall_score(y_true, y_pred, zero_division=0),
            "auc":        roc_auc_score(y_true, y_prob),
            "mcc":        matthews_corrcoef(y_true, y_pred),
            "fpr":        fp / (fp + tn + 1e-9),
            "fp":         int(fp),
            "train_time": round(train_time, 2),
            "epoch_time": round(epoch_time, 3),
        })
        print(f"  seed {seed} | AUC {seed_results[-1]['auc']:.4f} | F1 {seed_results[-1]['f1']:.4f}")

    avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
    avg["model"] = label
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Classical ML Evaluation (RF, XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classical(clf, data, train_mask, test_mask, label):
    print(f"\n[ML]  {label} ...")
    X = data.x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = {
        "model":      label,
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "auc":        roc_auc_score(y_test, y_prob),
        "mcc":        matthews_corrcoef(y_test, y_pred),
        "fpr":        float(fp / (fp + tn + 1e-9)),
        "fp":         int(fp),
        "train_time": round(train_time, 2),
        "epoch_time": 0.0,
    }
    print(f"  AUC {result['auc']:.4f} | F1 {result['f1']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Graph Neural Fraud Engine — Final Benchmark")
    print("=" * 60)

    data, train_mask, test_mask = load_data()
    IN = data.x.size(1)

    results = []

    # ── GNN Models ──────────────────────────────────────────────
    results.append(evaluate_gnn(EllipticGCN,   {"in_channels": IN, "hidden": 64},                          data, train_mask, test_mask, "EllipticGCN"))
    results.append(evaluate_gnn(EllipticGAT,   {"in_channels": IN, "hidden_channels": 64, "heads": 2},     data, train_mask, test_mask, "EllipticGAT"))
    results.append(evaluate_gnn(SAGE,           {"in_channels": IN, "hidden_channels": 64},                 data, train_mask, test_mask, "GraphSAGE"))
    results.append(evaluate_gnn(ResidualSAGE,   {"in_channels": IN, "hidden_channels": 32},                 data, train_mask, test_mask, "ResidualSAGE"))
    results.append(evaluate_gnn(TemporalSAGE,   {"in_channels": IN},                                        data, train_mask, test_mask, "TemporalSAGE"))
    results.append(evaluate_gnn(CAGNN,          {"in_channels": IN, "hidden_channels": 64},                 data, train_mask, test_mask, "CAGNN (Ours)"))

    # ── Classical Baselines ─────────────────────────────────────
    results.append(evaluate_classical(
        RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
        data, train_mask, test_mask, "Random Forest"
    ))
    results.append(evaluate_classical(
        XGBClassifier(n_estimators=100, scale_pos_weight=POS_WEIGHT,
                      eval_metric="logloss", random_state=42, n_jobs=-1),
        data, train_mask, test_mask, "XGBoost"
    ))

    # ── Save Results ────────────────────────────────────────────
    df = pd.DataFrame(results)
    col_order = ["model", "auc", "f1", "precision", "recall", "mcc", "fpr", "fp", "train_time", "epoch_time"]
    df = df[col_order].sort_values("auc", ascending=False)
    df.to_csv("results/final_benchmark_all_models.csv", index=False)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved to results/final_benchmark_all_models.csv")
