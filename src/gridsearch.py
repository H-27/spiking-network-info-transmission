import csv
import datetime as dt
import json
import os
import sys

# Use a non-interactive backend so figures can be saved from scripts
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve project root and ensure src is on sys.path
CURRENT_DIR = os.path.abspath(os.getcwd())
if os.path.basename(CURRENT_DIR).lower() == "src":
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    PROJECT_ROOT = CURRENT_DIR
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(fig: plt.Figure, out_path: str) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_current_fig(out_path: str) -> None:
    fig = plt.gcf()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_base_config(node_type: str) -> dict:
    # Prefer baseline configs if present; otherwise fall back to existing ones.
    if node_type.upper() == "LIF":
        candidates = [
            os.path.join(PROJECT_ROOT, "configs", "baseline_lif.json"),
        ]
    else:
        candidates = [
            os.path.join(PROJECT_ROOT, "configs", "baseline_izhikevich.json"),
            os.path.join(PROJECT_ROOT, "configs", "izhikevich_base.json"),
        ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                cfg = json.load(f)
            return cfg
    raise FileNotFoundError(
        f"No config file found for node_type={node_type}. Tried: {candidates}"
    )


def build_weights(cfg: dict, lateral_mode: str) -> tuple:
    """Construct feedforward and lateral weight matrices based on cfg and mode.
    Returns (ff1, rec1, ff2, rec2, ff3, rec3) as numpy arrays.
    """
    from connections import (
        connect_clustered_lateral,
        connect_one_to_one,
        connect_random,
    )

    n_in = cfg["n_inputs"]
    n1 = cfg["n_layer_one"]
    n2 = cfg["n_layer_two"]
    n3 = cfg["n_layer_three"]
    w_str = cfg.get("weight_strength", 1.0)

    # Feedforward: one-to-one
    ff1 = connect_one_to_one(n_in, n1)
    ff2 = connect_one_to_one(n1, n2)
    ff3 = connect_one_to_one(n2, n3)

    p_noise = cfg.get("noise_level", 0.1)

    if lateral_mode.lower() == "clustered":
        n_clusters = int(cfg.get("cluster_size", max(2, int(np.sqrt(n1)))))
        p_intra = float(cfg.get("cluster_p_intra", 0.8))
        rec1 = (
            connect_clustered_lateral(
                n1,
                n_clusters=n_clusters,
                p_intra=p_intra,
                connect_clusters=True,
                exclude_self=True,
            )
            + connect_random(n1, n1, p=p_noise)
        ) * w_str
        n_clusters2 = int(cfg.get("cluster_size", max(2, int(np.sqrt(n2)))))
        rec2 = (
            connect_clustered_lateral(
                n2,
                n_clusters=n_clusters2,
                p_intra=p_intra,
                connect_clusters=True,
                exclude_self=True,
            )
            + connect_random(n2, n2, p=p_noise)
        ) * w_str
        n_clusters3 = int(cfg.get("cluster_size", max(2, int(np.sqrt(n3)))))
        rec3 = (
            connect_clustered_lateral(
                n3,
                n_clusters=n_clusters3,
                p_intra=p_intra,
                connect_clusters=True,
                exclude_self=True,
            )
            + connect_random(n3, n3, p=p_noise)
        ) * w_str
    else:
        rec1 = connect_random(n1, n1, p=p_noise) * w_str
        rec2 = connect_random(n2, n2, p=p_noise) * w_str
        rec3 = connect_random(n3, n3, p=p_noise) * w_str

    return ff1, rec1, ff2, rec2, ff3, rec3


def generate_input(
    T: int, n_in: int, rate: float = 0.1, amplitude: float = 10.0
) -> torch.Tensor:
    """Bernoulli spike input of shape [T, n_in]."""
    prob = min(max(rate, 0.0), 1.0)
    return torch.bernoulli(torch.full((T, n_in), prob)).float() * amplitude


def run_experiment(
    node_type: str,
    weight_value: float,
    lateral_mode: str,
    out_dir: str,
    summary_writer=None,
) -> dict:
    # Import here to avoid module-level import after non-import code and to allow dynamic sys.path
    from networks import SpikingNetwork

    # Load and customize config
    cfg = load_base_config(node_type)
    cfg["node_type"] = node_type
    cfg["weight_strength"] = float(weight_value)
    cfg["cluster_size"] = (
        int(cfg.get("cluster_size", 0))
        if lateral_mode.lower() != "clustered"
        else max(2, int(np.sqrt(cfg["n_layer_one"])))
    )

    # Build network
    net = SpikingNetwork(cfg)
    ff1, rec1, ff2, rec2, ff3, rec3 = build_weights(cfg, lateral_mode)
    net.add_weights(ff1, rec1, ff2, rec2, ff3, rec3)
    net.build()

    # Inputs and run
    T = int(cfg.get("time", net.T))  # fall back to net.T
    n_in = cfg["n_inputs"]
    input_rate = float(cfg.get("input_rate", 0.1))
    amplitude = float(cfg.get("input_amplitude", 10.0))
    inp = generate_input(T, n_in, input_rate, amplitude)

    noise_mean = float(cfg.get("noise_mean", 0.0))
    noise_scale = float(cfg.get("noise_scale", 0.01))
    net.run(inp, input_mean=noise_mean, input_scale=noise_scale)

    # Metrics
    metrics = {}
    for L in (1, 2, 3):
        r, m, M, s = net.calculate_metrics(L)
        metrics[f"layer{L}"] = {
            "rsync": float(r),
            "mean_spike_count": float(m),
            "max_spike_count": float(M),
            "sparseness": float(s),
        }

    # Save run config + metrics
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save overview figure (all layers)
    try:
        net.plot_last(cfg)
        save_current_fig(os.path.join(out_dir, "overview.png"))
    except Exception as e:
        print(f"Warning: failed to save overview figure: {e}")

    # Save per-layer metrics bars and trends
    try:
        from networks import SpikingNetwork  # only to appease some static analyzers

        fig, _ = net.plot_metrics_all()
        save_fig(fig, os.path.join(out_dir, "metrics_all_layers.png"))
    except Exception as e:
        print(f"Warning: failed to save metrics_all_layers: {e}")

    for name in ["rsync", "mean", "max", "sparseness"]:
        try:
            fig, _ = net.plot_metric_trend(name, kind="line")
            save_fig(fig, os.path.join(out_dir, f"trend_{name}.png"))
        except Exception as e:
            print(f"Warning: failed to save trend {name}: {e}")

    # Append summary CSV row if writer provided
    if summary_writer is not None:
        row = {
            "node_type": node_type,
            "weight_strength": weight_value,
            "lateral_mode": lateral_mode,
            **{f"{k}_{m}": v for k, d in metrics.items() for m, v in d.items()},
        }
        summary_writer.writerow(row)

    return metrics


def main():
    # Grid search parameters
    NODE_TYPES = ["LIF"]  # ["LIF", "Izhikevich"] if both supported
    WEIGHT_VALUES = [0.2, 0.5, 1.0]
    LATERAL_CONNECTIONS = ["Random", "Clustered"]

    # Seeds for reproducibility
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Output directory
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_out = os.path.join(PROJECT_ROOT, "data", "gridsearch", timestamp)
    ensure_dir(root_out)

    # Create a summary CSV
    summary_csv = os.path.join(root_out, "summary.csv")
    fieldnames = [
        "node_type",
        "weight_strength",
        "lateral_mode",
        "layer1_rsync",
        "layer1_mean_spike_count",
        "layer1_max_spike_count",
        "layer1_sparseness",
        "layer2_rsync",
        "layer2_mean_spike_count",
        "layer2_max_spike_count",
        "layer2_sparseness",
        "layer3_rsync",
        "layer3_mean_spike_count",
        "layer3_max_spike_count",
        "layer3_sparseness",
    ]
    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for node in NODE_TYPES:
            for w in WEIGHT_VALUES:
                for lat in LATERAL_CONNECTIONS:
                    run_name = f"{node}_w{w}_lat{lat}"
                    out_dir = os.path.join(root_out, run_name)
                    print(f"Running: {run_name}")
                    run_experiment(node, w, lat, out_dir, summary_writer=writer)
                    print(f"Saved results to {out_dir}")

    print(f"All results saved under {root_out}")


if __name__ == "__main__":
    main()
