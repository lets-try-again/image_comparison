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
    def select_anchors(self, x_train: np.array, y_train: np.array):
        pass

    @abstractmethod
    def select_pairs(self, n: int, x_train: np.array, y_train: np.array):
        pass


class RandomSelectionPolicy(PairSelectionPolicy):

    def __init__(self, n_classes):
        self.pairs = dict()
        self.n_classes = n_classes

    def select_anchors(self, x_train: np.array, y_train: np.array) -> dict:
        """ Selects randomly one array from each class and calls it an anchor """

        for cl in range(self.n_classes):
            x_cl = x_train[y_train == cl]
            anchor_ind = np.random.choice(x_cl.shape[0])
            anchor = x_cl[anchor_ind]
            anchors[cl] = anchor
        return anchors

    def select_pairs(self, n: int, x_train: np.array, y_train: np.array):
        """ Choose 180 different images from different classes
        and 180 similar images from the same class"""

        anchors = self.select_anchors(x_train, y_train)

        for cl, anchor in anchors.items():
            x_pos = x_train[y_train == cl]
            x_neg = x_train[y_train != cl]

            positive_inds = np.random.choice(x_pos.shape[0], n)
            negative_inds = np.random.choice(x_neg.shape[0], n)
            positives = x_pos[positive_inds]
            negatives = x_neg[negative_inds]
            self.pairs[anchor] = (cl, positives, negatives)

        return self.pairs


class KLSelectionPolicy(PairSelectionPolicy):

    def calculate_KL_divergence(self, anchor):
        pass

    def select_anchors(self, x_train, y_train):
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

    from preprocessing.loader import get_mnist

    x_train, y_train, x_test, y_test = get_mnist()

    rsp = RandomSelectionPolicy(n_classes=10)
    anchors = rsp.select_anchors(x_train, y_train)
    pairs = rsp.select_pairs(200, x_train, y_train)
