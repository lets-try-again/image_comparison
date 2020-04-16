import numpy as np
from sklearn.manifold import TSNE


def reduce_dim(x: np.array) -> np.array:
    x_out = TSNE().fit_transform(x)
    return x_out
