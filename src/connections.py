import numpy as np


def create_diagonal_connection(n_inputs, n_neurons, weight=0.5, diagonal_width=1):
    """
    Create a diagonal connection matrix.

    :param n_inputs: Number of input neurons.
    :param n_neurons: Number of target neurons.
    :param weight: Weight to assign to the diagonal connections.
    :return: A numpy array representing the connection weights.
    """
    assert n_inputs == n_neurons, (
        "Number of inputs must equal number of neurons for diagonal connection."
    )
    w = np.zeros((n_inputs, n_neurons), dtype=np.float32)
    if diagonal_width > 1:
        for i in range(-diagonal_width // 2 + 1, diagonal_width // 2 + 1):
            np.fill_diagonal(w[max(0, i) :, max(0, -i) :], weight)
    else:
        np.fill_diagonal(w, weight)
    return w


def create_clustered_connection(
    n_neurons, cluster_size, window_size=1, density=1.0, weight=1.0
):
    """
    Create a clustered connection matrix.
    :param n_neurons: Total number of neurons.
    :param cluster_size: Size of each cluster.
    :param window_size: Size of the window for connections within clusters.
    :param weight: Weight to assign to the clustered connections.
    :return: A numpy array representing the connection weights.
    """
    assert n_neurons % cluster_size == 0, (
        "Total number of neurons must be divisible by cluster size."
    )

    num_clusters = n_neurons // cluster_size

    print(f"Number of clusters: {num_clusters}")
    w = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    for i in range(num_clusters + 1):
        print(f"Processing cluster {i}/{num_clusters}")
        start = i * cluster_size
        end = start + cluster_size
        centre = (start + end) // 2
        # Adjust window for edge clusters
        if i == 0 and window_size > 1:
            end += window_size // 2
            w[start:end, start:end] = weight
        # Adjust window for last cluster
        if i == num_clusters and window_size > 1:
            start -= window_size // 2
            w[start:end, start:end] = weight
        # Middle clusters
        if i >= 1 and i < num_clusters - 1 and window_size > 1:
            start -= window_size // 2
            end += window_size // 2
            w[start:end, start:end] = weight
        else:
            w[start:end, start:end] = weight

    return w


def create_clustered_connection_with_density(
    n_neurons,
    num_clusters,
    weight=1.0,
    window_size=1,
    density=1.0,
    weight_probability=1.0,
    distance_scale=5.0,
    distance_metric: str = "manhattan",
    normalize_prob: bool = True,
    rng: np.random.Generator | None = None,
):
    """Create a clustered connection matrix with distance-dependent probabilistic weights.

    For each cluster, a (possibly extended) window around the cluster is considered. Within that
    window, synaptic weights are set to `weight` with probability that decays with distance from the
    cluster center. The probability at the center is `weight_probability`, and decays as:

        p(d) = weight_probability * exp(- d / distance_scale)

    Distance can be computed with either a 1D Manhattan (|i-center| + |j-center|) or a 2D radial
    approximation (sqrt((i-center)^2 + (j-center)^2)) depending on `distance_metric`.

    Final sampling probability matrix P is (optionally normalized to max 1) and then scaled by
    `density` (global sparsity factor) and clipped to [0,1]. A Bernoulli sample decides if a weight
    is placed.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons (square matrix assumed).
    cluster_size : int
        Size of each (non-overlapping) base cluster.
    window_size : int, default 1
        Extra neurons (total span) to include around each cluster (half applied on each side).
        If 1, no extension (just cluster). If >1, the effective window expands.
    density : float, default 1.0
        Global multiplicative factor on probability (sparsity control). 1.0 keeps raw probabilities.
    weight_probability : float, default 1.0
        Base probability at the cluster center before distance decay.
    weight : float, default 1.0
        Weight value to assign when a synapse is instantiated.
    distance_scale : float, default 5.0
        Decay length scale; larger produces slower decay (broader connectivity).
    distance_metric : {'manhattan','radial'}
        Form of distance used in probability decay.
    normalize_prob : bool, default True
        If True, probability profile is normalized so the central peak = weight_probability.
    rng : np.random.Generator | None
        Optional NumPy Generator for reproducibility.

    Returns
    -------
    w : np.ndarray (n_neurons, n_neurons)
        Generated weight matrix.
    """
    cluster_size = n_neurons // num_clusters
    if rng is None:
        rng = np.random.default_rng()

    w = np.zeros((n_neurons, n_neurons), dtype=np.float32)

    half_extra = max(0, window_size // 2)

    for i in range(num_clusters):
        base_start = i * cluster_size
        base_end = base_start + cluster_size
        center = (base_start + base_end) // 2

        # Extended window bounds (clipped to matrix)
        start_w = max(0, base_start - half_extra)
        end_w = min(n_neurons, base_end + half_extra)

        idx_window = np.arange(start_w, end_w)
        # 1D distances from center
        dist_line = np.abs(idx_window - center)
        # Probability profile along one dimension
        p_line = weight_probability * np.exp(-dist_line / max(distance_scale, 1e-6))
        if normalize_prob and p_line.max() > 0:
            p_line = p_line / p_line.max() * weight_probability

        # 2D probability matrix
        if distance_metric == "manhattan":
            # Outer sum of distances
            D = dist_line[:, None] + dist_line[None, :]
        elif distance_metric == "radial":
            # Approx radial by combining squared distances
            D = np.sqrt(dist_line[:, None] ** 2 + dist_line[None, :] ** 2)
        else:
            raise ValueError("distance_metric must be 'manhattan' or 'radial'")

        P = weight_probability * np.exp(-D / max(distance_scale, 1e-6))
        if normalize_prob and P.max() > 0:
            P = P / P.max() * weight_probability

        # Global sparsity control
        P = np.clip(P * density, 0.0, 1.0)

        # Sample Bernoulli for this window
        mask = rng.random(P.shape) < P
        w[start_w:end_w, start_w:end_w][mask] = weight

    return w

def connect_one_to_one(n_pre, n_post, exclude_self=None):
    """
    1 on the main diagonal (i -> i), 0 elsewhere.
    Extra rows/cols stay 0 if sizes differ.
    """
    A = np.zeros((n_pre, n_post), dtype=np.float32)
    np.fill_diagonal(A, 1)
    return A

def connect_all_to_all(n_pre, n_post, exclude_self=True):
    """
    All ones. If square and exclude_self=True, zero the diagonal.
    """
    A = np.ones((n_pre, n_post), dtype=np.float32)
    if exclude_self and n_pre == n_post:
        np.fill_diagonal(A, 0)
    return A

def connect_random(n_pre, n_post, p=0.1, exclude_self=True):
    """
    Each edge present with prob p (independent).
    """
    A = (np.random.rand(n_pre, n_post) < p).astype(np.float32)
    if exclude_self and n_pre == n_post:
        np.fill_diagonal(A, 0)
    return A

def connect_distance(n_neurons, _, sigma=3.0, p_max=0.5, circular=False, exclude_self=True):
    """
    Distance-dependent binary connectivity (1D).
    - Neurons are placed on a 1D line with indices 0..n-1.
    - Connection prob decays with distance: p_ij = p_max * exp(-(d_ij^2)/(2*sigma^2))
    - If circular=True, distance wraps around (ring topology).

    Computes:
        A   : [n, n] binary adjacency (0/1)
        P   : [n, n] connection probabilities used
        D   : [n, n] pairwise distances

    Returns:
        A   : [n, n] binary adjacency (0/1)
    """
    idx = np.arange(n_neurons)
    D = np.abs(idx[:, None] - idx[None, :])  # |i-j|
    if circular:
        D = np.minimum(D, n_neurons - D)     # wrap-around distance on a ring

    # Pprobabilities decay exponentially with distance
    P = p_max * np.exp(-(D**2) / (2.0 * sigma**2))

    # Sample connections
    A = (np.random.rand(n_neurons, n_neurons) < P).astype(np.float32)

    if exclude_self:
        np.fill_diagonal(A, 0)
        P[np.diag_indices(n_neurons)] = 0.0

    return A
