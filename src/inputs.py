import numpy as np

def poisson_input(
    T, n_neurons, dt,
    target_idx,
    target_rate,
    bg_rate,
    batch_size=1
):
    """
    Generate Poisson spike trains in numpy.

    Args:
        T             : number of time steps
        n_neurons     : number of neurons
        dt            : timestep (ms)
        target_idx    : list of target neuron indices
        target_rate   : firing rate for target neurons (Hz)
        bg_rate       : firing rate for all other neurons (Hz)
        batch_size    : number of batches

    Returns:
        spikes : NumPy array [T, batch_size, n_neurons] of 0/1
    """
    # baseline rates
    rates = np.full(n_neurons, bg_rate, dtype=np.float32)
    rates[np.array(target_idx, dtype=int)] = target_rate

    # probability per step
    p = rates * (dt / 1000.0)  # Hz × s
    p = np.clip(p, 0.0, 1.0)

    # sample
    rand = np.random.rand(T, batch_size, n_neurons)
    spikes = (rand < p).astype(np.float32)
    return spikes

def generate_input(T, n_inputs, dt, target_rate=10, bg_rate=5):
    # Stimulate a group of neurons noisily
    target_idx = np.concatenate(
        [np.arange(0, 14), np.arange(52, 76), np.arange(80, 100), np.arange(200, 300)]
    )

    poisson_spikes = poisson_input(
        T, n_inputs, dt, target_idx, target_rate, bg_rate, batch_size=1
    )

    return poisson_spikes



