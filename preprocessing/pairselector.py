import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class PairSelectionPolicy(ABC):
    """ Triplet selection policy """

    @abstractmethod
    def select_anchor(self, x_train, y_train):
        pass

    @abstractmethod
    def select_pair(self, anchor):
        pass


class RandomSelectionPolicy(PairSelectionPolicy):

    def select_anchor(self, x_train, y_train):
        pass

    def select_pair(self, anchor):
        pass


class KLSelectionPolicy(PairSelectionPolicy):

    def calculate_KL_divergence(self, anchor):
        pass

    def select_anchor(self, x_train, y_train):
        pass

    def select_pair(self, anchor):
        pass


"""
If there are N = 60,000 train images for 10 classes,
then there're 6,000 images for one class, which makes
(6,000 * 5,999 / 2) * 54,000 possible triplet combinations

To reduce the number. we'll do:
- choose just one anchor (6000 -> 10)
- for each anchor we calculate a metrics between it and every other point
(let's say KL divergence, but it will later be the prediction of our network)
"""


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist()
    print(x_train.shape)
    print(x_test.shape)