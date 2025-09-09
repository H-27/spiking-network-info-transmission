import matplotlib.pyplot as plt


def visualize_weight_matrix(matrix, title="Weight Connection Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Post-synaptic Neuron")
    plt.ylabel("Pre-synaptic Neuron")
    plt.show()


def plot_network_activity(
    inputs,
    connectivity,
    voltage,
    spikes,
    title=None,
    sub_titles=["Input spikes", "Feedforward connectivity", "Voltage", "Spikes"],
):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

    h, w = voltage.shape
    ax[0].imshow(inputs)
    ax[0].set_title(sub_titles[0])
    ax[0].set_xlabel("Time [ms]")
    ax[0].set_ylabel("Input neurons")
    ax[0].set_aspect(w / h)
    ax[0].invert_yaxis()

    ax[1].imshow(connectivity)
    ax[1].set_title(sub_titles[1])
    ax[1].set_xlabel("Post")
    ax[1].set_ylabel("Pre")
    ax[1].invert_yaxis()

    h, w = voltage.shape
    ax[2].imshow(voltage)
    ax[2].set_title(sub_titles[2])
    ax[2].set_xlabel("Time [ms]")
    ax[2].set_ylabel("Neurons")
    ax[2].set_aspect(w / h)
    ax[2].invert_yaxis()

    ax[3].imshow(spikes)
    ax[3].imshow(spikes)
    ax[3].set_title(sub_titles[3])
    ax[3].set_xlabel("Time [ms]")
    ax[3].set_ylabel("Neurons")
    ax[3].set_aspect(w / h)
    ax[3].invert_yaxis()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
