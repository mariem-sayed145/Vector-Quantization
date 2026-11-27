import numpy as np


def initialize_centroids(
    vectors: np.ndarray,
    k: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Initialize centroids using K-Means++.
    """

    rng = np.random.default_rng(random_state)

    n_samples = len(vectors)

    centroids = np.empty(
        (k, vectors.shape[1]),
        dtype=np.float32
    )

    first_index = rng.integers(n_samples)

    centroids[0] = vectors[first_index]

    distances = np.sum(
        (vectors - centroids[0]) ** 2,
        axis=1
    )

    for i in range(1, k):

        total_distance = distances.sum()

    if total_distance == 0:
        next_index = rng.integers(n_samples)
    else:
        probabilities = distances / total_distance
        next_index = rng.choice(
            n_samples,
            p=probabilities
    )

        centroids[i] = vectors[next_index]

        new_distances = np.sum(
            (vectors - centroids[i]) ** 2,
            axis=1
        )

        distances = np.minimum(
            distances,
            new_distances
        )

    return centroids


def find_nearest_centroids(
    vectors: np.ndarray,
    centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest centroid for each vector.

    Returns:
        indices and distances.
    """

    distances = np.sum(
        (vectors[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2
    )

    indices = np.argmin(distances, axis=1)

    minimum_distances = distances[
        np.arange(len(vectors)),
        indices
    ]

    return indices, minimum_distances


def update_centroids(
    vectors: np.ndarray,
    indices: np.ndarray,
    k: int,
    distances: np.ndarray
) -> np.ndarray:
    """
    Update centroids using the mean of assigned vectors.

    Empty clusters are reinitialized using vectors
    with the largest assignment errors.
    """

    dimensions = vectors.shape[1]

    centroids = np.zeros(
        (k, dimensions),
        dtype=np.float32
    )

    counts = np.bincount(
        indices,
        minlength=k
    )

    np.add.at(centroids, indices, vectors)

    non_empty = counts > 0

    centroids[non_empty] /= counts[non_empty, None]

    empty_clusters = np.where(~non_empty)[0]

    if len(empty_clusters) > 0:

        farthest_indices = np.argsort(distances)[::-1]

        for cluster_index, vector_index in zip(
            empty_clusters,
            farthest_indices
        ):
            centroids[cluster_index] = vectors[vector_index]

    return centroids


def generate_codebook(
    vectors: np.ndarray,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate a codebook using K-Means clustering.
    """

    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Vectors must be a non-empty 2D array.")

    if k < 1 or k > len(vectors):
        raise ValueError("K must be between 1 and the number of vectors.")

    centroids = initialize_centroids(
        vectors,
        k,
        random_state
    )

    for _ in range(max_iterations):

        indices, distances = find_nearest_centroids(
            vectors,
            centroids
        )

        new_centroids = update_centroids(
            vectors,
            indices,
            k,
            distances
        )

        movement = np.linalg.norm(
            new_centroids - centroids
        )

        centroids = new_centroids

        if movement < tolerance:
            break

    return centroids