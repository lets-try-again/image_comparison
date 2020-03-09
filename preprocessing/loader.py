import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from preprocessing.pairselector import PairSelectionPolicy


def get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


class DataLoader():
    """
    This class will feed the data to neural network
    """
    def __init__(self, pair_policy: PairSelectionPolicy):
        pass


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist()
