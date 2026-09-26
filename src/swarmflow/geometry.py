import numpy as np


def limit_norm(vector, limit):
    return vector * np.minimum(1., limit / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12))


def swept_distance(relative_start, relative_end):
    """Minimum center distance on a linear relative-motion segment."""
    delta = relative_end - relative_start
    fraction = np.clip(-np.sum(relative_start * delta, axis=-1) /
                       np.maximum(np.sum(delta * delta, axis=-1), 1e-15), 0, 1)
    return np.linalg.norm(relative_start + fraction[..., None] * delta, axis=-1)
