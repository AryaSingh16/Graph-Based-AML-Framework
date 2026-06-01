import pandas as pd
import json
from pathlib import Path

def export_model_comparison_txt(csv_path="results/performance_benchmark.csv", out_path="results/model_comparison_table.txt"):
    """Generates a formal, aligned TXT report for model performance."""
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} does not exist. Skipping TXT export.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Add status columns for the report
    df['Explainability'] = 'Yes'
    df['Production Ready'] = df['Model'].apply(lambda x: 'Yes' if 'Residual' in x or 'SAGE' in x else 'No')
    
    # Format the table string
    with open(out_path, "w") as f:
        f.write("Transaction Intelligence Framework - Model Comparison Report\n")
        f.write("="*70 + "\n\n")
        f.write(df.to_string(index=False, justify='left'))
        f.write("\n\n" + "="*70 + "\n")
        f.write("Generated automatically by Graffe Engine Analysis Tool.\n")
    print(f"Model comparison report exported to {out_path}")

def export_system_efficiency_txt(json_path="results/efficiency_metrics.json", out_path="results/system_efficiency_table.txt"):
    """Generates a formal, aligned TXT report for system efficiency."""
    if not Path(json_path).exists():
        print(f"Warning: {json_path} does not exist. Skipping TXT export.")
        return
        
    with open(json_path, "r") as f:
        metrics = json.load(f)
        
    with open(out_path, "w") as f:
        f.write("Transaction Intelligence Framework - System Efficiency Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Metric':<35} | {'Value':<15}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Full Graph Latency':<35} | {metrics['full_graph_latency_ms']:>10.2f} ms\n")
        f.write(f"{'Local Subgraph Latency (Avg)':<35} | {metrics['subgraph_latency_ms_avg']:>10.2f} ms\n")
        f.write(f"{'Engine Speedup Ratio':<35} | {metrics['speedup']:>10.1f}x\n")
        f.write(f"{'Representative CPU Peak RAM':<35} | {metrics['peak_ram_mb']:>10.1f} MB\n")
        f.write(f"{'Inference Mode':<35} | {'CPU / Local Neighborhood':>27}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Note: Benchmarked on Elliptic (203k nodes) using commodity CPU.\n")
    print(f"System efficiency report exported to {out_path}")

if __name__ == "__main__":
    export_model_comparison_txt()
    export_system_efficiency_txt()
