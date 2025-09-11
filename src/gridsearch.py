import os
from logging import config

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
    # Define params
    T = config["time"]
    dt = config["dt"]
    n_neurons = config["n_neurons"]
    n_inputs = config["n_inputs"]
    target_rate = 30.0
    bg_rate = 30.0
