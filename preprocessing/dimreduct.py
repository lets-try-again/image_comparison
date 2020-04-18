import numpy as np
from sklearn.manifold import TSNE


def reduce_dim(x: np.array) -> np.array:
    print(f'Shape of input of t-SNE: {x.shape}')
    x_out = TSNE().fit_transform(x)
    print(f'Shape of output of t-SNE: {x_out.shape}')
    return x_out
