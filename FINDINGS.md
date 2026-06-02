# Research Findings: 

This document details the critical discoveries made during the systematic evaluation of Graph Neural Networks (GNNs) versus Classical Machine Learning on the Elliptic Bitcoin dataset.

## Finding 1: The "Graph Camouflage" Phenomenon
**Hypothesis:** GNNs will outperform feature-only models because fraudulent transactions operate in coordinated network rings.  
**Result:** False. Random Forest (0.9946 AUC) outperformed the best standard GNN (0.8699 AUC) by 12.5%.
**Root Cause:** The Elliptic dataset exhibits extreme **heterophily** for illicit nodes. Fraudsters actively engage in *Graph Camouflage*—routing dirty Bitcoin through high-volume, legitimate services (exchanges, mixers) to blend in. Standard GNNs (GCN, GraphSAGE) act as low-pass filters; they average the target's features with its legitimate neighbors, effectively "washing away" the suspicious signal.

## Finding 2: CAGNN (Camouflage-Aware GNN) Efficacy
**Hypothesis:** A GNN specifically designed to detect heterophily will outperform standard spatial GNNs.
**Result:** True. By introducing a dual-branch architecture (low-pass for homophily, high-pass for heterophily) governed by a learnable gate, **CAGNN** successfully amplified the "differences" between a camouflaged fraudster and its clean neighbors, recovering significant predictive power lost by standard SAGE models.

## Finding 3: Velocity Features Were a Dead End
**Hypothesis:** Engineered structural features (Burst Score, Fan-out Ratio, TTL Proxy) will improve baseline accuracy by capturing high-velocity money laundering topologies.
**Result:** False. Ablation studies proved these features contributed zero predictive power.
**Conclusion:** On the Elliptic dataset, the purely local financial traits of the transaction (e.g., raw amount, fees, size) contain the vast majority of the signal. Topological metrics were redundant to what the base 165 features already captured.

## Finding 4: Threshold Optimization Cannot Fix Poor Calibration
**Hypothesis:** The high False Positive rate of ResidualSAGE (3,380 FPs) can be fixed by shifting the decision threshold from 0.5 to 0.95.
**Result:** False. FPs only reduced to 2,995, and the F1 score remained abysmal (0.367).
**Conclusion:** The model suffered from severe probability miscalibration, not a thresholding error. It lacked the necessary discriminative boundary to separate camouflaged fraud from true legitimate behavior.

## Final Production Strategy
Rather than forcing a GNN to act as the primary classifier (which would compromise accuracy for the sake of using deep learning), the optimal production strategy is a **Hybrid Architecture**:
1. **Classical ML (Random Forest/XGBoost)** acts as the high-speed, high-accuracy scoring engine.
2. **Graph Neural Networks (CAGNN / Subgraph Sampling)** act as the Explainability Engine, retrieving the 1-hop subgraph and highlighting suspicious neighborhood flows to provide human-readable context for the ML prediction.
