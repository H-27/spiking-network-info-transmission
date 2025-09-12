import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use a non-interactive backend before any pyplot import occurs anywhere
import matplotlib

matplotlib.use("Agg")

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

# Progress bar (graceful fallback if tqdm not installed)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    class _DummyTqdm:
        def __init__(self, total=None, desc=None, unit=None):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, total=None, desc=None, unit=None):
        if iterable is None:
            return _DummyTqdm(total=total, desc=desc, unit=unit)
        return iterable


from connections import connect_clustered_lateral, connect_one_to_one, connect_random
from inputs import generate_input
from networks import SpikingNetwork


def plot_metrics(input_type: str):
    """Aggregate saved metrics for a given input type and plot comparisons.

    Expects files in results/<input_type>/ named like:
      metrics_weight{weight}_connection{conn_type}_input{input_type}.json
    with structure: {"layer1": {...}, "layer2": {...}, "layer3": {...}}

    Saves one figure per metric with subplots per connection type.
    """
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

    # Desired order, include Baseline as well
    desired_conns = ["Clustered", "Random", "Baseline"]
    present_conns = [c for c in desired_conns if c in conns]
    if not present_conns:
        return

    for metric_name in sorted(metric_names):
        fig, axes = plt.subplots(
            1, len(present_conns), figsize=(5 * len(present_conns), 4), sharey=True
        )
        # Ensure axes is iterable
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax, conn in zip(axes, present_conns):
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
        axes[-1].legend(loc="best")
        fig.suptitle(f"{metric_name} vs weight — {input_type}")
        out_path = os.path.join(base_dir, f"{metric_name}_trend_{input_type}.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    # Grid search parameters
    NODE_TYPE = "LIF"  # "LIF" or "Izhikevich"
    WEIGHT_VALUES = [0.2, 0.5, 1.0]
    LATERAL_CONNECTIONS = ["Baseline", "Clustered", "Random"]

    # Load config
    # Determine project root (parent of notebooks directory)
    current_dir = os.path.abspath(os.getcwd())
    # If running from project root, notebooks/ may not be in path; handle both cases
    if os.path.basename(current_dir).lower() == "src":
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if NODE_TYPE == "LIF":
        config_path = os.path.join(project_root, "configs", "baseline_lif.json")
    elif NODE_TYPE == "Izhikevich":
        config_path = os.path.join(project_root, "configs", "baseline_izhikevich.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Define params
    T = config["time"]
    dt = config["dt"]
    n_inputs = config["n_inputs"]

    # Create feedforward connections
    feed_forward_upscale = 20
    ff1 = (
        connect_one_to_one(config["n_inputs"], config["n_layer_one"])
        * feed_forward_upscale
    )
    ff2 = (
        connect_one_to_one(config["n_layer_one"], config["n_layer_two"])
        * feed_forward_upscale
    )
    ff3 = (
        connect_one_to_one(config["n_layer_two"], config["n_layer_three"])
        * feed_forward_upscale
    )
    input_factor = 10
    random_recurrent_factor = 2.5

    # Create inputs
    """
    T, n_inputs, dt, target_rate=10, bg_rate=5
    """
    # Weakly distributed input

    target_weakly_distr = np.concatenate(
        [
            np.arange(0, 14),
            np.arange(52, 76),
            np.arange(80, 100),
            np.arange(200, 300),
        ]
    )
    weakly_distributed = (
        generate_input(
            T=T,
            n_inputs=n_inputs,
            dt=dt,
            target_rate=15,
            bg_rate=5,
        )
    ) * input_factor
    weakly_distributed = torch.from_numpy(weakly_distributed[:, 0, :])

    # Strongly distributed input
    target_strongly_distr = np.concatenate(
        [
            np.arange(0, 14),
            np.arange(52, 76),
            np.arange(80, 100),
            np.arange(200, 300),
        ]
    )
    strongly_distributed = (
        generate_input(
            T=T,
            n_inputs=n_inputs,
            dt=dt,
            target_rate=50,
            bg_rate=5,
        )
    ) * input_factor
    strongly_distributed = torch.from_numpy(strongly_distributed[:, 0, :])

    # Weakly clustered input
    target_weakly_clustered = np.concatenate(
        [
            np.arange(0, 29),
            np.arange(60, 89),
            np.arange(120, 149),
            np.arange(270, 299),
        ]
    )
    weakly_clustered = (
        generate_input(
            T=T,
            n_inputs=n_inputs,
            dt=dt,
            target_rate=15,
            bg_rate=5,
        )
    ) * input_factor
    weakly_clustered = torch.from_numpy(weakly_clustered[:, 0, :])

    # Strongly clustered input
    target_strongly_clustered = np.concatenate(
        [
            np.arange(0, 29),
            np.arange(60, 89),
            np.arange(120, 149),
            np.arange(270, 299),
        ]
    )
    strongly_clustered = (
        generate_input(
            T=T,
            n_inputs=n_inputs,
            dt=dt,
            target_rate=50,
            bg_rate=5,
        )
        * input_factor
    )
    strongly_clustered = torch.from_numpy(strongly_clustered[:, 0, :])

    inputs = {
        "weakly_distributed": weakly_distributed,
        "strongly_distributed": strongly_distributed,
        "weakly_clustered": weakly_clustered,
        "strongly_clustered": strongly_clustered,
    }

    total_runs = len(WEIGHT_VALUES) * len(LATERAL_CONNECTIONS) * len(inputs)
    pbar = tqdm(total=total_runs, desc="Grid search", unit="run")

    for weight in WEIGHT_VALUES:
        for conn_type in LATERAL_CONNECTIONS:
            for input_type, input_data in inputs.items():
                net = SpikingNetwork(config)
                # Create connections
                if conn_type == "Clustered":
                    rec1 = (
                        connect_clustered_lateral(
                            config["n_layer_one"],
                            n_clusters=config["cluster_size"],
                            p_intra=0.8,
                            connect_clusters=True,
                            exclude_self=True,
                        )
                        * weight
                        + connect_random(
                            config["n_layer_one"],
                            config["n_layer_one"],
                            p=config["noise_level"],
                        )
                        * weight
                    )
                    rec2 = (
                        connect_clustered_lateral(
                            config["n_layer_one"],
                            n_clusters=config["cluster_size"],
                            p_intra=0.8,
                            connect_clusters=True,
                            exclude_self=True,
                        )
                        * weight
                        + connect_random(
                            config["n_layer_two"],
                            config["n_layer_two"],
                            p=config["noise_level"],
                        )
                        * weight
                    )
                    rec3 = (
                        connect_clustered_lateral(
                            config["n_layer_one"],
                            n_clusters=config["cluster_size"],
                            p_intra=0.8,
                            connect_clusters=True,
                            exclude_self=True,
                        )
                        * weight
                        + connect_random(
                            config["n_layer_three"],
                            config["n_layer_three"],
                            p=config["noise_level"],
                        )
                        * weight
                    )
                elif conn_type == "Random":
                    rec1 = (
                        connect_random(
                            config["n_layer_one"],
                            config["n_layer_one"],
                            p=config["noise_level"] * random_recurrent_factor,
                        )
                        * weight
                    )
                    rec2 = (
                        connect_random(
                            config["n_layer_two"],
                            config["n_layer_two"],
                            p=config["noise_level"] * random_recurrent_factor,
                        )
                        * weight
                    )
                    rec3 = (
                        connect_random(
                            config["n_layer_three"],
                            config["n_layer_three"],
                            p=config["noise_level"] * random_recurrent_factor,
                        )
                        * weight
                    )
                else:
                    # Baseline: empty recurrent connections (no lateral)
                    rec1 = (
                        connect_random(
                            config["n_layer_one"],
                            config["n_layer_one"],
                            p=config["noise_level"],
                        )
                    ) * 0
                    rec2 = (
                        connect_random(
                            config["n_layer_two"],
                            config["n_layer_two"],
                            p=config["noise_level"],
                        )
                    ) * 0
                    rec3 = (
                        connect_random(
                            config["n_layer_three"],
                            config["n_layer_three"],
                            p=config["noise_level"],
                        )
                    ) * 0

                net.add_weights(ff1, rec1, ff2, rec2, ff3, rec3)
                net.build()

                # Run simulation
                net.run(inputs=input_data)
                # Save results
                results_dir = os.path.join(project_root, "results")
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                # Compute metrics for all layers
                metrics_all = {}
                for i in (1, 2, 3):
                    rsync, mean_sc, max_sc, sparseness, rate_of_fire = (
                        net.calculate_metrics(layer=i)
                    )
                    metrics_all[f"layer{i}"] = {
                        "rsync": rsync,
                        "mean_sc": mean_sc,
                        "max_sc": max_sc,
                        "sparseness": sparseness,
                        "rate_of_fire": rate_of_fire,
                    }
                # Create a subdirectory per input type to keep filenames short & valid
                run_dir = os.path.join(results_dir, input_type)
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)

                result_metrics_path = os.path.join(
                    run_dir,
                    f"metrics_weight{weight}_connection{conn_type}_input{input_type}.json",
                )
                result_plots_path = os.path.join(
                    run_dir,
                    f"plots_weight{weight}_connection{conn_type}_input{input_type}.png",
                )

                # Write metrics json
                with open(result_metrics_path, "w") as f:
                    json.dump(metrics_all, f, indent=2)

                net.plot_last(config, save_to=result_plots_path)
                print(
                    f"Completed: weight={weight}, conn={conn_type}, input={input_type}"
                )
                pbar.update(1)

    pbar.close()

    # After all runs, produce comparison plots per input type
    for it in inputs.keys():
        plot_metrics(it)
