# Spiking Network Information Transmission

This project investigates information flow in multi-layer spiking neural networks using BindsNET. It compares purely feedforward networks against networks with lateral (within-layer) connectivity and measures how different structures affect spike propagation and coding.

## Repository structure

- `src/`
  - `gridsearch.py` — runs experiments across connection types (Baseline, Clustered, Random) and weight scales; saves plots and metrics.
  - `networks.py` — defines the `SpikingNetwork` class (layers, connections, monitors, plotting, metrics).
  - `connections.py` — utilities to create feedforward and lateral connectivity patterns (one-to-one, random, clustered, etc.).
  - `inputs.py` — Poisson input generators and helper to build input tensors.
  - `utils.py` — metric functions: synchrony, mean/max spike count, sparsity, firing rate.
  - `functions.py` — legacy helpers (not used by gridsearch).
  - `demo_model.ipynb` — example notebook (exploration/demo).
- `configs/`
  - `baseline_lif.json` — baseline configuration for LIF-based networks.
  - `baseline_izhikevich.json` — baseline configuration for Izhikevich-based networks.
- `results/` — output directory populated by gridsearch: per-input subfolders with metrics JSON and plot PNGs.
- `notebooks/` — exploratory notebooks (`network_display.ipynb`, `testnetwork.ipynb`).
- `pyproject.toml` — Python project metadata and dependencies.
- `uv.lock` — lockfile for uv package manager.
- `LICENSE` — project license.

## Dependencies

- Python 3.12+
- NumPy, SciPy, Matplotlib
- PyTorch, torchvision
- BindsNET (installed from GitHub in `pyproject.toml`)

## Setup

Using uv (recommended):
1) Install uv: https://github.com/astral-sh/uv
2) From the project root, install dependencies:

```
uv sync
```

Using pip:

```
pip install -e .
```

## Running experiments

Run the grid search from the project root:

```
uv run .\src\gridsearch.py
```

This will:
- generate inputs
- build networks for each combination of weight ∈ [0.2, 0.5, 1.0], connection ∈ {Baseline, Clustered, Random}
- run simulations
- save metrics JSON and plots to `results/<input_type>/...`

A tqdm progress bar will be shown (falls back gracefully if tqdm isn’t installed).

## Notes

- Baseline uses zero lateral connections.
- Random uses random lateral connections (probability controlled by `noise_level`).
- Clustered uses structured lateral connectivity with optional inter-cluster inhibition.
- Adjust feedforward strength and input scaling in `src/gridsearch.py`.
- Thresholds and neuron model are set via the config files; LIF and Izhikevich nodes are supported.
- Presentation assets: some figures/files used in the presentation were created by hand using the notebook `notebooks/testnetwork.ipynb` and may not be produced directly by the grid search script.
