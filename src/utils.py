import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, Iterable


def visualize_weight_matrix(matrix, title="Weight Connection Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Post-synaptic Neuron")
    plt.ylabel("Pre-synaptic Neuron")
    plt.show()
