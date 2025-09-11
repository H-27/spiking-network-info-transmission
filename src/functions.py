def generate_input(target_rate=10, bg_rate=5):
    # Stimulate a group of neurons noisily
    target_idx = np.concatenate(
        [np.arange(0, 14), np.arange(52, 76), np.arange(80, 100), np.arange(200, 300)]
    )

    poisson_spikes = poisson_input(
        T, n_inputs, dt, target_idx, target_rate, bg_rate, batch_size=1
    )

    return poisson_spikes


def connect_clustered_lateral(
    n_neurons,
    n_clusters=None,
    cluster_size=None,
    p_intra=0.8,
    evenly_spaced=True,
    connect_clusters=True,
    exclude_self=True,
):
    """
    This function creates a clustered connectivity matrix.
    Neurons should be organized in clusters, with higher connection probability within clusters than between clusters.
    Args:
        n_neurons       : number of neurons
        n_clusters     : number of clusters (if None, computed from cluster_size)
        cluster_size   : size of each cluster (if None, computed from n_clusters)
        p_intra        : probability of connection within a cluster
        p_inter        : probability of connection between clusters
        evenly_spaced  : if True, clusters are evenly spaced; if False, clusters are random
        connect_clusters: if True, connect clusters with p_inter; if False, only intra-cluster connections
        exclude_self   : if True, no self-connections
    """
    p_inter = 0.4  # probability of connection between neurons in different clusters
    p_connect_cluster = 0.33  # probability of inhibitory connection between clusters

    if n_clusters is None and cluster_size is None:
        n_clusters = int(np.sqrt(n_neurons))
    if cluster_size is None:
        cluster_size = n_neurons // n_clusters
    if n_clusters is None:
        n_clusters = n_neurons // cluster_size

    A = np.zeros((n_neurons, n_neurons), dtype=np.float32)

    # Determine cluster indices
    if evenly_spaced:
        cluster_indices = [
            list(range(i * cluster_size, (i + 1) * cluster_size))
            for i in range(n_clusters)
        ]
    else:
        all_indices = np.arange(n_neurons)
        np.random.shuffle(all_indices)
        cluster_indices = [
            all_indices[i * cluster_size : (i + 1) * cluster_size].tolist()
            for i in range(n_clusters)
        ]

    # Intra-cluster connections
    for indices in cluster_indices:
        for i in indices:
            for j in indices:
                if i != j or not exclude_self:
                    if np.random.rand() < p_intra:
                        A[i, j] = np.random.rand() * 0.8 + 0.2

    # Inhibitory connections between clusters
    if connect_clusters:
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if np.random.rand() < p_connect_cluster:
                    for pre in cluster_indices[i]:
                        for post in cluster_indices[j]:
                            A[pre, post] = -(np.random.rand())
                            A[post, pre] = -(
                                np.random.rand()
                            )  # symmetric inhibitory connection

    # Random connections
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                for pre in cluster_indices[i]:
                    for post in cluster_indices[j]:
                        if np.random.rand() < p_inter:
                            # random between -1 and 1
                            A[pre, post] = np.random.rand() * 2 - 1

    # Exclude self-connections if specified
    if exclude_self:
        np.fill_diagonal(A, 0)

    return A
