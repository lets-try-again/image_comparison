import numpy as np
from sklearn.preprocessing import MinMaxScaler


def shrink(x_train, x_test):
    # normalize
    n = x_train.shape[0] + x_test.shape[0]
    X = np.concatenate([x_train, x_test]).reshape((n, -1))
    X_scaled = MinMaxScaler().fit_transform(X).reshape((n, 2, 28, 28, 1))
    x_train, x_test = X_scaled[:x_train.shape[0]], X_scaled[x_train.shape[0]:]
    return x_train, x_test
