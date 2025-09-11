import matplotlib.pyplot as plt
import torch
from typing import Dict, Optional, Iterable
from .network import Network
from .monitors import AbstractMonitor


def visualize_weight_matrix(matrix, title="Weight Connection Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Post-synaptic Neuron")
    plt.ylabel("Pre-synaptic Neuron")
    plt.show()



class InputMonitor(AbstractMonitor):
    # language=rst
    """
    Records the direct inputs provided to layers during network simulation.
    """

    def __init__(
        self,
        network: "Network",
        layers: Iterable[str],
        time: Optional[int] = None,
        device: str = "cpu",
    ):
        # language=rst
        """
        Constructs an ``InputMonitor`` object.

        :param network: The network object.
        :param layers: Iterable of strings indicating names of layers to record inputs for.
        :param time: If not ``None``, pre-allocate memory for input recording.
        :param device: The device to store the recordings on.
        """
        super().__init__()
        self.network = network
        self.layers = layers
        self.time = time
        self.device = device

        if self.time is None:
            self.device = "cpu"

        self.recording = {layer: [] for layer in self.layers}
        self.clean = True

    def record(self, inputs: Dict[str, torch.Tensor]) -> None:
        # language=rst
        """
        Records the current batch of inputs for the specified layers.

        :param inputs: The dictionary of inputs provided to the network's ``run()`` method.
        """
        self.clean = False
        for layer in self.layers:
            if layer in inputs:
                data = inputs[layer].unsqueeze(0)
                
                # Check if layer input matches network's layer size
                if hasattr(self.network.layers.get(layer), 'n_units') and data.shape[-1] != self.network.layers[layer].n_units:
                    print(f"Warning: Input shape {data.shape} for layer '{layer}' does not match layer size {self.network.layers[layer].n_units}. Monitoring may be inaccurate.")
                    
                self.recording[layer].append(
                    torch.empty_like(data, device=self.device, requires_grad=False).copy_(
                        data, non_blocking=True
                    )
                )

                if self.time is not None and len(self.recording[layer]) > self.time:
                    self.recording[layer].pop(0)

    def get(self, layer: str) -> torch.Tensor:
        # language=rst
        """
        Return the recorded inputs for a specific layer.

        :param layer: The name of the layer to get the input recording for.
        :return: Tensor of shape ``[time, batch_size, n_units]``.
        """
        if self.clean or layer not in self.recording or not self.recording[layer]:
            return torch.empty(0, device=self.device)
        else:
            return_logs = torch.cat(self.recording[layer], 0)
            if self.time is None:
                self.recording[layer] = []  # Clear logs if not pre-allocated
            return return_logs

    def reset(self) -> None:
        # language=rst
        """
        Resets all input recordings to empty lists.
        """
        self.recording = {layer: [] for layer in self.layers}
        self.clean = True