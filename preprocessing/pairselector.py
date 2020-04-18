from abc import ABC, abstractmethod
from typing import Dict

# import warnings
# warnings.filterwarnings("ignore")
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian
from itertools import combinations, product


class PairSelectionPolicy(ABC):
    """ Pair selection policy """

    @abstractmethod
    def select_pairs(self, n: int, x: np.array, y: np.array):
        pass


class RandomSelectionPolicy(PairSelectionPolicy):

    def __init__(self, n_classes, random_state=None):
        self.pairs = dict()
        self.n_classes = n_classes
        self.random_state = random_state

    def get_number_of_classes(self, y: np.array):
        return np.unique(y).shape[0]

    def select_anchors(self, x: np.array, y: np.array) -> dict:
        """ Selects randomly one array from each class and calls it an anchor """
        anchors = dict()

        # check that we passed a correct number of classes
        if self.n_classes == -1:
            self.n_classes = self.get_number_of_classes(y)
        else:
            assert self.get_number_of_classes(y) == self.n_classes

        for cl in range(self.n_classes):
            x_cl = x[y == cl]
            np.random.seed(self.random_state)
            anchor_ind = np.random.choice(x_cl.shape[0])
            anchor = x_cl[anchor_ind]
            anchors[cl] = anchor
        return anchors

    def select_pairs(self, n: int, x: np.array, y: np.array):
        """ Choose n different images from different classes
        and n similar images from the same class"""

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


class AllWithAllPolicy(PairSelectionPolicy):

    def __init__(self):
        self.groups_by_label = []
        self.all_pairs = []
        self.all_targets = []

    def select_pairs(self, n: int, x: np.array, y: np.array):

        for c in range(int(np.max(y))+1):
            xa = x[y == c][:n]
            self.groups_by_label.append(xa)

        self.create_pos_pairs()
        self.create_neg_pairs()

        x, y = np.concatenate(self.all_pairs), np.concatenate(self.all_targets)
        return x, y

    def create_pos_pairs(self):
        for group in self.groups_by_label:
            pairs = self.all_combinations(group)
            self.all_pairs.append(pairs)
            self.all_targets.append([1 for _ in range(pairs.shape[0])])

    def create_neg_pairs(self):
        for i, group in enumerate(self.groups_by_label):
            if i < len(self.groups_by_label) - 1:
                groups_in_front = np.concatenate(self.groups_by_label[i + 1:])
                pairs = self.all_combinations(group, groups_in_front)
                self.all_pairs.append(pairs)
                self.all_targets.append([0 for _ in range(pairs.shape[0])])

    def all_combinations(self, x: np.array, y=None):
        """
        Returns pairs of images provided arrays with images
        if y is None, returns all pairs within x
        if y is provided, returns all pairs where the first element
            is from x, and the second element is from y
        """
        if y is not None:
            return np.array(list(product(x, y)))
        return np.array(list(combinations(x, 2)))


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

    # from preprocessing.loader import get_mnist
    #
    # x_train, y_train, x_test, y_test = get_mnist()
    #
    # rsp = RandomSelectionPolicy(n_classes=10)
    # pairs = rsp.select_pairs(5, x_train, y_train)

    y = np.array([[0, 4, 5]])
    z = np.array([[9, 9, 9]])

    a = np.array([[999, 999, 999]])
    b = np.array([[888, 888, 888]])

    x = np.concatenate([y, z], axis=0)
    c = np.concatenate([a, b], axis=0)

    print(x.shape)
    out = np.array(list(product(x, c)))
    print(out)
    print(out.shape)
