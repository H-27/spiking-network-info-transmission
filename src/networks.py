import numpy as np
import matplotlib.pyplot as plt
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, AdaptiveLIFNodes, Input, IzhikevichNodes 
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.learning import WeightDependentPostPre, Hebbian, MSTDPET
from connections import connect_random
from typing import Dict, Iterable, Optional, Type
from utils import measure_rsync, measure_mean_sc, measure_max_sc, measure_sparseness

class SpikingNetwork(Network):
    def __init__(self, config):
        """Spiking neural network with three hidden layers.
        Parameters
        ----------
        config : dict
        """
        super(SpikingNetwork, self).__init__()
        self.n_inputs = config["n_inputs"]
        self.n_layer_one = config["n_layer_one"]
        self.n_layer_two = config["n_layer_two"]
        self.n_layer_three = config["n_layer_three"]
        self.time = config["time"]
        self.dt = config["dt"]
        self.device = config["device"]
        self.simulation_length = config["simulation_length"]
        self.T = int(self.simulation_length / self.dt)

        self.learning_lateral_type = config["learning_lateral"]
        if self.learning_lateral_type == "WeightDependentPostPre":
            self.learning_lateral = WeightDependentPostPre
        elif self.learning_lateral_type == "Hebbian":
            self.learning_lateral = Hebbian
        elif self.learning_lateral_type == "MSTDPET":
            self.learning_lateral = MSTDPET
        else:
            print("No learning for lateral connections.")
            self.learning_lateral = None
        self.learning_lateral_nu = config["learning_lateral_nu"]

        self.learning_feedforward_type = config["learning_feedforward"]
        if self.learning_feedforward_type == "WeightDependentPostPre":
            self.learning_feedforward = WeightDependentPostPre
        elif self.learning_feedforward_type == "Hebbian":
            self.learning_feedforward = Hebbian
        elif self.learning_feedforward_type == "MSTDPET":
            self.learning_feedforward = MSTDPET
        else:
            print("No learning for feedforward connections.")
            self.learning_feedforward = None
        self.learning_feedforward_nu = config["learning_feedforward_nu"]
        
        self.node_type = config["node_type"]
        self.input_node = Input(n=self.n_inputs)
        if self.node_type == "LIF":
            self.node_class = LIFNodes
        elif self.node_type == "AdaptiveLIF":
            self.node_class = AdaptiveLIFNodes
        elif self.node_type == "Izhikevich":
            self.node_class = IzhikevichNodes
        else:
            raise ValueError(f"Unsupported node type: {self.node_type}")

    def add_weights(self, ff1, rec1, ff2, rec2, ff3, rec3):
        """Add weight matrices for the network connections.

        Parameters
        ----------
        ff1 : array-like
            Feedforward weights from input to layer one.
        rec1 : array-like
            Recurrent weights within layer one.
        ff2 : array-like
            Feedforward weights from layer one to layer two.
        rec2 : array-like
            Recurrent weights within layer two.
        ff3 : array-like
            Feedforward weights from layer two to layer three.
        rec3 : array-like
            Recurrent weights within layer three.
        """
        self.ff1_weights = torch.from_numpy(ff1)
        self.rec1_weights = torch.from_numpy(rec1)
        self.ff2_weights = torch.from_numpy(ff2)
        self.rec2_weights = torch.from_numpy(rec2)
        self.ff3_weights = torch.from_numpy(ff3)
        self.rec3_weights = torch.from_numpy(rec3)
    
    def build(self):
        """Build the spiking neural network architecture."""
        self.network = Network(dt=self.dt)#, device=self.device)
        self.input_layer = Input(n=self.n_inputs, traces=True)
        self.layer_one = self.node_class(n=self.n_layer_one, traces=True)
        self.layer_two = self.node_class(n=self.n_layer_two, traces=True)
        self.layer_three = self.node_class(n=self.n_layer_three, traces=True)
        self.network.add_layer(self.input_layer, name="Input")
        self.network.add_layer(self.layer_one, name="Layer1")
        self.network.add_layer(self.layer_two, name="Layer2")
        self.network.add_layer(self.layer_three, name="Layer3")
        self.network.add_connection(
            Connection(source=self.input_layer, target=self.layer_one, w=self.ff1_weights, update_rule=self.learning_feedforward, nu=self.learning_feedforward_nu), 
            source="Input", target="Layer1"
        )
        self.network.add_connection(
        Connection(source=self.layer_one, target=self.layer_two, w=self.ff2_weights, update_rule=self.learning_feedforward, nu=self.learning_feedforward_nu), 
        source="Layer1", target="Layer2"
        )
        self.network.add_connection(
        Connection(source=self.layer_two, target=self.layer_three, w=self.ff3_weights, update_rule=self.learning_feedforward, nu=self.learning_feedforward_nu), 
        source="Layer2", target="Layer3"
        )
        self.network.add_connection(
            Connection(source=self.layer_one, target=self.layer_one, w=self.rec1_weights, update_rule=self.learning_lateral, nu=self.learning_lateral_nu), 
            source="Layer1", target="Layer1"
        )
        self.network.add_connection(
            Connection(source=self.layer_two, target=self.layer_two, w=self.rec2_weights, update_rule=self.learning_lateral, nu=self.learning_lateral_nu), 
            source="Layer2", target="Layer2"
        )
        self.network.add_connection(
            Connection(source=self.layer_three, target=self.layer_three, w=self.rec3_weights, update_rule=self.learning_lateral, nu=self.learning_lateral_nu), 
            source="Layer3", target="Layer3"
        )

        self.input_monitor = Monitor(self.input_layer, state_vars=["s"], time=self.T)
        self.layer_one_monitor = Monitor(self.layer_one, state_vars=["s", "v"], time=self.T)
        self.layer_two_monitor = Monitor(self.layer_two, state_vars=["s", "v"], time=self.T)
        self.layer_three_monitor = Monitor(self.layer_three, state_vars=["s", "v"], time=self.T)
        self.network.add_monitor(self.input_monitor, name="InputMonitor")
        self.network.add_monitor(self.layer_one_monitor, name="Layer1Monitor")
        self.network.add_monitor(self.layer_two_monitor, name="Layer2Monitor")
        self.network.add_monitor(self.layer_three_monitor, name="Layer3Monitor")

    def run(self, inputs, input_mean=0.0, input_scale=0.01):
        """Run the network on provided inputs."""
        # Old noise implementation
        noise1 = torch.from_numpy(np.random.normal(loc=input_mean, scale=input_scale, size=(self.T, self.n_layer_one))).float()
        noise2 = torch.from_numpy(np.random.normal(loc=input_mean, scale=input_scale, size=(self.T, self.n_layer_two))).float()
        noise3 = torch.from_numpy(np.random.normal(loc=input_mean, scale=input_scale, size=(self.T, self.n_layer_three))).float()
        print(inputs.shape, noise1.shape, noise2.shape, noise3.shape)
        self.network.run(inputs={"Input": inputs,
                                "Layer1": noise1,
                                "Layer2": noise2,
                                "Layer3": noise3
                                }, time=self.T, train=True)

    def calculate_metrics(self, layer: int = 1):
        """Calculate and print activity metrics for a selected hidden layer.

        Parameters
        ----------
        layer : int
            Which layer to analyze (1, 2, or 3).
        """
        assert layer in (1, 2, 3), "layer must be 1, 2, or 3"
        # Choose layer monitors
        if layer == 1:
            layer_mon = self.layer_one_monitor
        elif layer == 2:
            layer_mon = self.layer_two_monitor
        else:
            layer_mon = self.layer_three_monitor
        spikes_t = layer_mon.get("s")
        if spikes_t.dim() == 3 and spikes_t.shape[1] == 1:
            spikes_t = spikes_t[:, 0, :]
        # Convert to (neurons, time)
        spikes_np = spikes_t.T.cpu().numpy().astype(float)
        # Calculate metrics
        
        rsync = measure_rsync(spikes_np)
        mean_sc = measure_mean_sc(spikes_np)
        max_sc = measure_max_sc(spikes_np)
        sparseness = measure_sparseness(spikes_np)
        return rsync, mean_sc, max_sc, sparseness
    
    def plot_last(self, config):
        """Plot activity for all layers."""
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
        # Retrieve monitors
        inp_s = self.input_monitor.get("s")
        inp_s = inp_s[:, 0, :]
        inp_s = inp_s.T.cpu().numpy().astype(float)  # [T, B, N_in] or [T, N_in]

        layer1_s = self.layer_one_monitor.get("s")
        layer1_s = layer1_s[:, 0, :]
        layer1_s = layer1_s.T.cpu().numpy().astype(float)

        layer1_v = self.layer_one_monitor.get("v")
        layer1_v = layer1_v[:, 0, :]
        layer1_v = layer1_v.T.cpu().numpy().astype(float)

        layer2_s = self.layer_two_monitor.get("s")
        layer2_s = layer2_s[:, 0, :]
        layer2_s = layer2_s.T.cpu().numpy().astype(float)

        layer2_v = self.layer_two_monitor.get("v")
        layer2_v = layer2_v[:, 0, :]
        layer2_v = layer2_v.T.cpu().numpy().astype(float)

        layer3_s = self.layer_three_monitor.get("s")
        layer3_s = layer3_s[:, 0, :]
        layer3_s = layer3_s.T.cpu().numpy().astype(float)

        layer3_v = self.layer_three_monitor.get("v")
        layer3_v = layer3_v[:, 0, :]
        layer3_v = layer3_v.T.cpu().numpy().astype(float)
        
        # plot input
        # Input
        h, w = layer1_v.shape
        ax[0, 0].imshow(inp_s, aspect=w / h, origin="lower")
        ax[0, 0].set_title("Input spikes")
        ax[0, 0].set_xlabel("Time [ms]")
        ax[0, 0].set_ylabel("Input neurons")
        # Input ff weights
        ff_np = self.ff1_weights.detach().cpu().numpy()
        ax[0, 1].imshow(ff_np, aspect="auto", origin="lower")
        ax[0, 1].set_title("Feedforward")
        ax[0, 1].set_xlabel("Post")
        ax[0, 1].set_ylabel("Pre")
        # Input rec weights
        rec_np = self.rec1_weights.detach().cpu().numpy()
        ax[0, 2].imshow(rec_np, aspect="auto", origin="lower")
        ax[0, 2].set_title("Recurrent")
        ax[0, 2].set_xlabel("Post")
        ax[0, 2].set_ylabel("Pre")
        # Input voltage
        h, w = layer1_v.shape
        ax[0, 3].imshow(layer1_v, aspect=w / h, origin="lower")
        ax[0, 3].set_title("Voltage")
        ax[0, 3].set_xlabel("Time [ms]")
        ax[0, 3].set_ylabel("Neurons")
        # Layer 1  output spikes
        ax[0, 4].imshow(layer1_s, aspect=w / h, origin="lower")
        ax[0, 4].set_title("Spikes")
        ax[0, 4].set_xlabel("Time [ms]")
        ax[0, 4].set_ylabel("Neurons")
        # Layer 2
        # Input
        h, w = layer2_v.shape
        noise = np.random.normal(loc=config["noise_mean"], scale=config["noise_scale"], size=(self.n_layer_two, self.T))
        ax[1, 0].imshow(layer1_s + noise, aspect=w / h, origin="lower")
        ax[1, 0].set_title("Layer 1 spikes + noise")
        ax[1, 0].set_xlabel("Time [ms]")
        ax[1, 0].set_ylabel("Neurons")
        # Layer 2 ff weights
        ff_np = self.ff2_weights.detach().cpu().numpy()
        ax[1, 1].imshow(ff_np, aspect="auto", origin="lower")
        ax[1, 1].set_title("Feedforward")
        ax[1, 1].set_xlabel("Post")
        ax[1, 1].set_ylabel("Pre")
        # Layer 2 rec weights
        rec_np = self.rec2_weights.detach().cpu().numpy()
        ax[1, 2].imshow(rec_np, aspect="auto", origin="lower")
        ax[1, 2].set_title("Recurrent")
        ax[1, 2].set_xlabel("Post")
        ax[1, 2].set_ylabel("Pre")
        # Layer 2 voltage
        h, w = layer2_v.shape
        ax[1, 3].imshow(layer2_v, aspect=w / h, origin="lower")
        ax[1, 3].set_title("Voltage")
        ax[1, 3].set_xlabel("Time [ms]")
        ax[1, 3].set_ylabel("Neurons")
        # Layer 2 output spikes
        ax[1, 4].imshow(layer2_s, aspect=w / h, origin="lower")
        ax[1, 4].set_title("Spikes")
        ax[1, 4].set_xlabel("Time [ms]")
        ax[1, 4].set_ylabel("Neurons")
        # Layer 3
        # Input
        h, w = layer3_v.shape
        noise = np.random.normal(loc=config["noise_mean"], scale=config["noise_scale"], size=(self.n_layer_three, self.T))
        ax[2, 0].imshow(np.add(layer2_s, noise), aspect=w / h, origin="lower")
        ax[2, 0].set_title("Layer 2 spikes + noise")
        ax[2, 0].set_xlabel("Time [ms]")
        ax[2, 0].set_ylabel("Neurons")
        # Layer 3 ff weights
        ff_np = self.ff3_weights.detach().cpu().numpy()
        ax[2, 1].imshow(ff_np, aspect="auto", origin="lower")
        ax[2, 1].set_title("Feedforward")
        ax[2, 1].set_xlabel("Post")
        ax[2, 1].set_ylabel("Pre")
        # Layer 3 rec weights
        rec_np = self.rec3_weights.detach().cpu().numpy()
        ax[2, 2].imshow(rec_np, aspect="auto", origin="lower")
        ax[2, 2].set_title("Recurrent")
        ax[2, 2].set_xlabel("Post")
        ax[2, 2].set_ylabel("Pre")
        # Layer 3 voltage
        h, w = layer3_v.shape
        ax[2, 3].imshow(layer3_v, aspect=w / h, origin="lower")
        ax[2, 3].set_title("Voltage")
        ax[2, 3].set_xlabel("Time [ms]")
        ax[2, 3].set_ylabel("Neurons")
        # Layer 3 output spikes
        ax[2, 4].imshow(layer3_s, aspect=w / h, origin="lower")
        ax[2, 4].set_title("Spikes")
        ax[2, 4].set_xlabel("Time [ms]")
        ax[2, 4].set_ylabel("Neurons")
