import warnings
from typing import List, Tuple
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf


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
        return x_train, y_train


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist()
