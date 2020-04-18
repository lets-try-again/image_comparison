import warnings
from typing import List, Tuple
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing.pairselector import *


def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])


class Ordering:
    """
    Takes pairs produced by pair selector
    Applies pairing and returns numpy arrays for training
    """

    @staticmethod
    def get_consecutive_pairs(pairs: dict) -> Tuple[np.array, np.array]:

        all_pairs = []
        y = []

        for key, values in pairs.items():
            anchor, positives, negatives = values
            all_pairs.extend([(anchor, positive) for positive in positives])
            all_pairs.extend([(anchor, negative) for negative in negatives])
            y.extend([1 for _ in range(len(positives))] + [0 for _ in range(len(negatives))])

        x_train, y_train = np.asarray(all_pairs, dtype='float32'), np.asarray(y, dtype='float32')
        print(f'All pairs shape {len(all_pairs)}')
        return x_train, y_train


def load_and_split(n_pairs=5000, n_classes=-1):
    # load the data
    x, y = get_mnist()

    # choose which classes to process:
    if n_classes > 0:
        print(f'Selecting {n_classes} classes')
        new_x = []
        new_y = []
        for cl in range(n_classes):
            print(f'For class {cl} found {x[y == cl].shape} shaped array')
            new_x.append(x[y == cl])
            new_y.append(y[y == cl])
        x = np.concatenate(new_x)
        y = np.concatenate(new_y)
        print(f'Total shape: {x.shape}')

    # select a pair policy
    # pairs = RandomSelectionPolicy(n_classes=n_classes, random_state=42).select_pairs(n_pairs, x, y)
    # x, y = Ordering.get_consecutive_pairs(pairs)

    x, y = AllWithAllPolicy().select_pairs(n_pairs, x, y)
    x = np.expand_dims(x, axis=4)
    return train_test_split(x, y, test_size=0.25)


if __name__ == '__main__':

    n_total_pairs = 10000
    n_pairs = int(np.sqrt(n_total_pairs / 2))
    x_train, x_test, y_train, y_test = load_and_split(n_pairs=50, n_classes=2)

    print(x_train.shape[0] + x_test.shape[0], 'pairs')

