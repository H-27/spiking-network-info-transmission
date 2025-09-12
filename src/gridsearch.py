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
import re

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
    # Import here to honor sys.path adjustment above
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


# ---------------- New Poisson input variants -----------------


def _to_torch_TN(x: np.ndarray) -> torch.Tensor:
    # x: [T, 1, N] -> [T, N]
    return torch.from_numpy(x[:, 0, :]).float()


def make_inputs(cfg: dict) -> dict:
    # Import here to honor sys.path adjustment above
    from inputs import poisson_input

    T = int(cfg.get("time", cfg.get("simulation_length", 0)))
    dt_ms = float(cfg.get("dt", 1.0))
    n_inputs = int(cfg["n_inputs"])
    input_factor = float(cfg.get("input_factor", 20.0))

    # Helper to clip indices within range
    def clip_idx(arr):
        arr = np.array(arr, dtype=int)
        return arr[(arr >= 0) & (arr < n_inputs)]

    # Index sets (clip to n_inputs)
    distr_idx = clip_idx(
        np.concatenate(
            [
                np.arange(0, 14),
                np.arange(52, 76),
                np.arange(80, 100),
                np.arange(200, 300),
            ]
        )
    )
    clust_idx = clip_idx(
        np.concatenate(
            [
                np.arange(0, 29),
                np.arange(60, 89),
                np.arange(120, 149),
                np.arange(270, 299),
            ]
        )
    )

    wd = (
        poisson_input(
            T, n_inputs, dt_ms, distr_idx, target_rate=15, bg_rate=5, batch_size=1
        )
        * input_factor
    )
    sd = (
        poisson_input(
            T, n_inputs, dt_ms, distr_idx, target_rate=50, bg_rate=5, batch_size=1
        )
        * input_factor
    )
    wc = (
        poisson_input(
            T, n_inputs, dt_ms, clust_idx, target_rate=15, bg_rate=5, batch_size=1
        )
        * input_factor
    )
    sc = (
        poisson_input(
            T, n_inputs, dt_ms, clust_idx, target_rate=50, bg_rate=5, batch_size=1
        )
        * input_factor
    )

    return {
        "weakly_distributed": _to_torch_TN(wd),
        "strongly_distributed": _to_torch_TN(sd),
        "weakly_clustered": _to_torch_TN(wc),
        "strongly_clustered": _to_torch_TN(sc),
    }


# ---------------- Plot aggregation like user's script -----------------


def plot_metrics_aggregate(project_root: str, input_type: str):
    base_dir = os.path.join(project_root, "results", input_type)
    if not os.path.isdir(base_dir):
        return

    pattern = re.compile(
        r"metrics_weight(?P<weight>[^_]+)_connection(?P<conn>[^_]+)_input"
    )

    # metrics[layer][metric][conn][weight] = value
    metrics = {f"layer{i}": {} for i in (1, 2, 3)}
    weights = set()
    conns = set()

    for fname in os.listdir(base_dir):
        if not (fname.startswith("metrics_") and fname.endswith(".json")):
            continue
        m = pattern.search(fname)
        if not m:
            continue
        try:
            weight = float(m.group("weight"))
        except Exception:
            continue
        conn = m.group("conn")
        conns.add(conn)
        weights.add(weight)
        fpath = os.path.join(base_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        for layer_key, layer_vals in data.items():
            for metric_name, val in layer_vals.items():
                metrics[layer_key].setdefault(metric_name, {}).setdefault(conn, {})[
                    weight
                ] = val

    if not any(metrics[layer] for layer in metrics):
        return

    weights = sorted(weights)
    conns = sorted(conns)

    layer_colors = {"layer1": "tab:blue", "layer2": "tab:orange", "layer3": "tab:green"}

    # Collect all metric names across layers
    metric_names = set()
    for layer in metrics.values():
        metric_names.update(layer.keys())

    for metric_name in sorted(metric_names):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, conn in zip(axes, ["Clustered", "Random"]):
            for layer_key in ("layer1", "layer2", "layer3"):
                by_conn = metrics.get(layer_key, {}).get(metric_name, {})
                y = [by_conn.get(conn, {}).get(w, np.nan) for w in weights]
                ax.plot(
                    weights,
                    y,
                    marker="o",
                    label=layer_key,
                    color=layer_colors[layer_key],
                )
            ax.set_title(conn)
            ax.set_xlabel("weight")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(metric_name)
        axes[1].legend(loc="best")
        fig.suptitle(f"{metric_name} vs weight — {input_type}")
        out_path = os.path.join(base_dir, f"{metric_name}_trend_{input_type}.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def run_experiment(
    node_type: str,
    weight_value: float,
    lateral_mode: str,
    input_name: str,
    inputs_tensor: torch.Tensor,
    out_dir: str,
    summary_writer=None,
) -> dict:
    # Import here to honor sys.path adjustment above
    from networks import SpikingNetwork

    # Load and customize config
    cfg = load_base_config(node_type)
    cfg["node_type"] = node_type
    cfg["weight_strength"] = float(weight_value)
    cfg["cluster_size"] = max(2, int(np.sqrt(cfg["n_layer_one"])))

    # Build network
    ff1, rec1, ff2, rec2, ff3, rec3 = build_weights(cfg, lateral_mode)
    net = SpikingNetwork(cfg)
    net.add_weights(ff1, rec1, ff2, rec2, ff3, rec3)
    net.build()

    # Run
    net.run(
        inputs_tensor,
        input_mean=float(cfg.get("noise_mean", 0.0)),
        input_scale=float(cfg.get("noise_scale", 0.01)),
    )

    # Metrics
    metrics = {}
    for L in (1, 2, 3):
        r, m, M, s, rof = net.calculate_metrics(L)
        metrics[f"layer{L}"] = {
            "rsync": float(r),
            "mean_sc": float(m),
            "max_sc": float(M),
            "sparseness": float(s),
            "rate_of_fire": float(rof),
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

    # Append summary CSV row if writer provided
    if summary_writer is not None:
        row = {
            "node_type": node_type,
            "weight_strength": weight_value,
            "lateral_mode": lateral_mode,
            "input_type": input_name,
            **{f"{k}_{m}": v for k, d in metrics.items() for m, v in d.items()},
        }
        summary_writer.writerow(row)

    return metrics


def main():
    # Grid search parameters
    NODE_TYPE = "LIF"
    WEIGHT_VALUES = [0.2, 0.5, 1.0]
    LATERAL_CONNECTIONS = ["Clustered", "Random"]

    # Seeds for reproducibility
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Output root
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root_out = os.path.join(PROJECT_ROOT, "data", "gridsearch", timestamp)
    ensure_dir(root_out)

    # Results folder like the user's script
    results_root = os.path.join(PROJECT_ROOT, "results")
    ensure_dir(results_root)

    # Summary CSV at root_out
    summary_csv = os.path.join(root_out, "summary.csv")
    fieldnames = [
        "node_type",
        "weight_strength",
        "lateral_mode",
        "input_type",
        "layer1_rsync",
        "layer1_mean_sc",
        "layer1_max_sc",
        "layer1_sparseness",
        "layer1_rate_of_fire",
        "layer2_rsync",
        "layer2_mean_sc",
        "layer2_max_sc",
        "layer2_sparseness",
        "layer2_rate_of_fire",
        "layer3_rsync",
        "layer3_mean_sc",
        "layer3_max_sc",
        "layer3_sparseness",
        "layer3_rate_of_fire",
    ]

    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        # Load base config once to build inputs
        base_cfg = load_base_config(NODE_TYPE)
        inputs_by_name = make_inputs(base_cfg)

        for w in WEIGHT_VALUES:
            for lat in LATERAL_CONNECTIONS:
                for input_name, input_tensor in inputs_by_name.items():
                    print(f"Running: weight={w}, conn={lat}, input={input_name}")

                    # Per-run directory inside gridsearch root
                    run_dir = os.path.join(
                        root_out, f"{NODE_TYPE}_w{w}_lat{lat}_{input_name}"
                    )

                    # Also save metrics/plots under results/<input_name>/ like user's naming
                    results_dir = os.path.join(results_root, input_name)
                    ensure_dir(results_dir)

                    metrics = run_experiment(
                        NODE_TYPE,
                        w,
                        lat,
                        input_name,
                        input_tensor,
                        run_dir,
                        summary_writer=writer,
                    )

                    # Write metrics json with requested filename pattern
                    metrics_path = os.path.join(
                        results_dir,
                        f"metrics_weight{w}_connection{lat}_input{input_name}.json",
                    )
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)

                    # Save plot image via plot_last already saved (overview.png). Duplicate to requested name.
                    plots_path = os.path.join(
                        results_dir,
                        f"plots_weight{w}_connection{lat}_input{input_name}.png",
                    )
                    try:
                        # Copy file if exists
                        src_plot = os.path.join(run_dir, "overview.png")
                        if os.path.exists(src_plot):
                            import shutil

                            shutil.copyfile(src_plot, plots_path)
                    except Exception as e:
                        print(f"Warning: failed to save plots to results folder: {e}")

    # After all runs, produce comparison plots per input type
    for input_name in (
        "weakly_distributed",
        "strongly_distributed",
        "weakly_clustered",
        "strongly_clustered",
    ):
        plot_metrics_aggregate(PROJECT_ROOT, input_name)

    print(f"All results saved under {root_out}")


if __name__ == "__main__":
    main()
