from abc import ABC, abstractmethod
from typing import Dict

# import warnings
# warnings.filterwarnings("ignore")
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PairSelectionPolicy(ABC):
    """ Pair selection policy """

    @abstractmethod
    def select_anchors(self, x: np.array, y: np.array):
        pass

    @abstractmethod
    def select_pairs(self, n: int, x: np.array, y: np.array):
        pass


class RandomSelectionPolicy(PairSelectionPolicy):

    def __init__(self, n_classes=10, random_state=None):
        self.pairs = dict()
        self.n_classes = n_classes
        self.random_state = random_state

    def select_anchors(self, x: np.array, y: np.array) -> dict:
        """ Selects randomly one array from each class and calls it an anchor """
        anchors = dict()

        for cl in range(self.n_classes):
            x_cl = x[y == cl]
            np.random.seed(self.random_state)
            anchor_ind = np.random.choice(x_cl.shape[0])
            anchor = x_cl[anchor_ind]
            anchors[cl] = anchor
        return anchors

    def select_pairs(self, n: int, x: np.array, y: np.array):
        """ Choose 180 different images from different classes
        and 180 similar images from the same class"""

        anchors = self.select_anchors(x, y)

        for cl, anchor in anchors.items():
            x_pos = x[y == cl]
            x_neg = x[y != cl]

            np.random.seed(self.random_state + 1)
            positive_inds = np.random.choice(x_pos.shape[0], n)
            np.random.seed(self.random_state + 1)
            negative_inds = np.random.choice(x_neg.shape[0], n)

            positives = x_pos[positive_inds]
            negatives = x_neg[negative_inds]
            self.pairs[cl] = (anchor, positives, negatives)

        return self.pairs


class KLSelectionPolicy(PairSelectionPolicy):

    def calculate_KL_divergence(self, anchor):
        pass

    def select_anchors(self, x_train, y_train):
        pass

    def select_pair(self, anchor):
        pass


class DeepOCSVMAnchorSelector:
    """
    Most normal and most anomalous in-class examples
    could determined  by anomaly detection
    """
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

    from preprocessing.loader import get_mnist

    x_train, y_train, x_test, y_test = get_mnist()

    rsp = RandomSelectionPolicy(n_classes=10)
    pairs = rsp.select_pairs(5, x_train, y_train)
