import os

from debugpy.common import json

from connections import connect_clustered_lateral, connect_one_to_one, connect_random
from inputs import generate_input
from networks import SpikingNetwork

if __name__ == "__main__":
    # Grid search parameters
    NODE_TYPE = "LIF"  # "LIF" or "Izhikevich"
    WEIGHT_VALUES = [0.2, 0.5, 1.0]
    LATERAL_CONNECTIONS = ["Clustered", "Random"]

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
    n_neurons = config["n_neurons"]
    n_inputs = config["n_inputs"]
    target_rate = 30.0
    bg_rate = 30.0

    # Create feedforward connections
    ff1 = connect_one_to_one(config["n_inputs"], config["n_layer_one"])
    ff2 = connect_one_to_one(config["n_layer_one"], config["n_layer_two"])
    ff3 = connect_one_to_one(config["n_layer_two"], config["n_layer_three"])

    # Create inputs
    weakly_distributed = generate_input(
        n_inputs=n_inputs,
        target_rate=target_rate,
        bg_rate=bg_rate,
        distribution="weakly",
    )
    strongly_distributed = generate_input(
        n_inputs=n_inputs,
        target_rate=target_rate,
        bg_rate=bg_rate,
        distribution="strongly",
    )
    weakly_clustered = generate_input(
        n_inputs=n_inputs,
        target_rate=target_rate,
        bg_rate=bg_rate,
        distribution="weakly",
        cluster=True,
    )
    strongly_clustered = generate_input(
        n_inputs=n_inputs,
        target_rate=target_rate,
        bg_rate=bg_rate,
        distribution="strongly",
        cluster=True,
    )

    inputs = {
        "weakly_distributed": weakly_distributed,
        "strongly_distributed": strongly_distributed,
        "weakly_clustered": weakly_clustered,
        "strongly_clustered": strongly_clustered,
    }

    for weight in WEIGHT_VALUES:
        for conn_type in LATERAL_CONNECTIONS:
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
            else:
                rec1 = (
                    connect_random(
                        config["n_layer_one"],
                        config["n_layer_one"],
                        p=config["noise_level"],
                    )
                    * weight
                )
                rec2 = (
                    connect_random(
                        config["n_layer_two"],
                        config["n_layer_two"],
                        p=config["noise_level"],
                    )
                    * weight
                )
                rec3 = (
                    connect_random(
                        config["n_layer_three"],
                        config["n_layer_three"],
                        p=config["noise_level"],
                    )
                    * weight
                )

            net.add_weights(ff1, rec1, ff2, rec2, ff3, rec3)
            net.build()

            # Run simulation
            net.run()
            # Save results
            results_dir = os.path.join(
                project_root,
                "results",
                f"gridsearch_{NODE_TYPE}_weight{weight}_{conn_type}",
            )
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            rsync, mean_sc, max_sc, sparseness, rate_of_fire = net.calculate_metrics()
            results = {
                "rsync": rsync,
                "mean_sc": mean_sc,
                "max_sc": max_sc,
                "sparseness": sparseness,
                "rate_of_fire": rate_of_fire,
            }
            result_metrics_path = os.path.join(
                results_dir,
                f"metrics_of_weight{weight}_connection{conn_type}_input.json",
            )
            result_plots_path = os.path.join(
                results_dir,
                f"plots_of_weight{weight}_connection{conn_type}_input.png",
            )

            net.plot_last(config, save_to=result_plots_path)
