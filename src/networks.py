import numpy as np
import matplotlib.pyplot as plt
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, AdaptiveLIFNodes, Input, IzhikevichNodes 
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.learning import WeightDependentPostPre, Hebbian, MSTDPET

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
        self.input_layer = Input(n=self.n_inputs)
        self.layer_one = self.node_class(n=self.n_layer_one)
        self.layer_two = self.node_class(n=self.n_layer_two)
        self.layer_three = self.node_class(n=self.n_layer_three)
        self.network.add_layer(self.input_layer, name="Input")
        self.network.add_layer(self.layer_one, name="Layer1")
        self.network.add_layer(self.layer_two, name="Layer2")
        self.network.add_layer(self.layer_three, name="Layer3")
        if self.learning_feedforward is not None:
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
        else:
            self.network.add_connection(
                Connection(source=self.input_layer, target=self.layer_one, w=self.ff1_weights), 
                source="Input", target="Layer1"
            )
            self.network.add_connection(
            Connection(source=self.layer_one, target=self.layer_two, w=self.ff2_weights), 
            source="Layer1", target="Layer2"
            )
            self.network.add_connection(
            Connection(source=self.layer_two, target=self.layer_three, w=self.ff3_weights), 
            source="Layer2", target="Layer3"
            )
        if self.learning_lateral is not None:
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
        else:
            self.network.add_connection(
                Connection(source=self.layer_one, target=self.layer_one, w=self.rec1_weights), 
                source="Layer1", target="Layer1"
            )
            self.network.add_connection(
                Connection(source=self.layer_two, target=self.layer_two, w=self.rec2_weights), 
                source="Layer2", target="Layer2"
            )
            self.network.add_connection(
                Connection(source=self.layer_three, target=self.layer_three, w=self.rec3_weights), 
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

    def run(self, inputs):
        """Run the network on provided inputs."""
        self.network.run(inputs={"Input": inputs}, time=self.T)

    def plot(self, layer: int = 1, title: str | None = None):
        """Plot activity for a selected hidden layer.

        Parameters
        ----------
        layer : int
            Which layer to visualize (1, 2, or 3).
        title : str | None
            Figure title.
        """
        assert layer in (1, 2, 3), "layer must be 1, 2, or 3"
        # Retrieve monitors
        inp_s = self.input_monitor.get("s")  # [T, B, N_in] or [T, N_in]
        if inp_s.dim() == 3 and inp_s.shape[1] == 1:
            inp_s = inp_s[:, 0, :]
        # Choose layer monitors & weights
        if layer == 1:
            layer_mon = self.layer_one_monitor
            ff_w = self.ff1_weights
            rec_w = self.rec1_weights
        elif layer == 2:
            layer_mon = self.layer_two_monitor
            ff_w = self.ff2_weights
            rec_w = self.rec2_weights
        else:
            layer_mon = self.layer_three_monitor
            ff_w = self.ff3_weights
            rec_w = self.rec3_weights
        spikes_t = layer_mon.get("s")
        volts_t = layer_mon.get("v")
        if spikes_t.dim() == 3 and spikes_t.shape[1] == 1:
            spikes_t = spikes_t[:, 0, :]
        if volts_t.dim() == 3 and volts_t.shape[1] == 1:
            volts_t = volts_t[:, 0, :]
        # Convert to (neurons, time)
        inputs_np = inp_s.T.cpu().numpy().astype(float)
        spikes_np = spikes_t.T.cpu().numpy().astype(float)
        volts_np = volts_t.T.cpu().numpy().astype(float)
        # Weights
        if isinstance(ff_w, torch.Tensor):
            ff_np = ff_w.detach().cpu().numpy()
        else:
            ff_np = np.asarray(ff_w, dtype=float)
        if isinstance(rec_w, torch.Tensor):
            rec_np = rec_w.detach().cpu().numpy()
        else:
            rec_np = np.asarray(rec_w, dtype=float)
        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(13, 3))
        h, w = volts_np.shape
        ax[0].imshow(inputs_np, aspect=w / h)
        ax[0].set_title("Input spikes")
        ax[0].set_xlabel("Time [ms]")
        ax[0].set_ylabel("Input neurons")
        ax[0].invert_yaxis()
        ax[1].imshow(ff_np)
        ax[1].set_title("Feedforward")
        ax[1].set_xlabel("Post")
        ax[1].set_ylabel("Pre")
        ax[1].invert_yaxis()
        ax[2].imshow(rec_np)
        ax[2].set_title("Lateral (Recurrent)")
        ax[2].set_xlabel("Post")
        ax[2].set_ylabel("Pre")
        ax[2].invert_yaxis()
        ax[3].imshow(volts_np, aspect=w / h)
        ax[3].set_title("Voltage")
        ax[3].set_xlabel("Time [ms]")
        ax[3].set_ylabel("Neurons")
        ax[3].invert_yaxis()
        ax[4].imshow(spikes_np, aspect=w / h)
        ax[4].set_title("Spikes")
        ax[4].set_xlabel("Time [ms]")
        ax[4].set_ylabel("Neurons")
        ax[4].invert_yaxis()
        if title is None:
            title = f"Layer {layer} Activity"
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        return fig, ax
