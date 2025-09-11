import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, Iterable
import numpy as np


def visualize_weight_matrix(matrix, title="Weight Connection Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Post-synaptic Neuron")
    plt.ylabel("Pre-synaptic Neuron")
    plt.show()

# From https://github.com/rnkblkv/snn_familiarity_saliency/blob/main/src/measure.py
def measure_rsync(firings):
    def exp_convolve(spike_train):
        tau = 5.0  # ms
        exp_kernel_time_steps = np.arange(0, tau*10, 1)
        decay = np.exp(-exp_kernel_time_steps/tau)
        exp_kernel = decay
        return np.convolve(spike_train, exp_kernel, 'same')  # 'valid'

    firings = np.apply_along_axis(exp_convolve, 1, firings)
    meanfield = np.mean(firings, axis=0) # spatial mean across cells, at each time
    variances = np.var(firings, axis=1)  # variance over time of each cell
    rsync = np.var(meanfield) / np.mean(variances)
    if np.isnan(rsync):
        rsync = 0.0
    return rsync

def measure_mean_sc(firings):
    return firings.sum(axis=1).mean() * 2

def measure_max_sc(firings):
    return firings.sum(axis=1).max() * 2

def measure_sparseness(firings):
    """
    This code calculates the lifetime sparseness of neural activity. 
    It's a measure of how selectively a neuron responds, 
    where a high sparseness value means the neuron is highly selective 
    (responds to a few specific stimuli) and a low value means it responds broadly to many stimuli."""
    n_neurons = firings.shape[0]
    if n_neurons < 2:
        return 0.0
    fr_mean = firings.mean()
    fr_sq_mean = (firings ** 2).mean()
    if fr_sq_mean == 0:
        return 0.0
    sparseness = (1 - (fr_mean ** 2) / fr_sq_mean) / (1 - 1 / n_neurons)
    return sparseness

