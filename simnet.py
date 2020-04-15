from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
# to run using CPU only
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split

from preprocessing.loader import get_mnist, Ordering
from preprocessing.pairselector import RandomSelectionPolicy
from utils.plot_loss import plot_loss


class Encoder(Model):

    """
    A network that finds a 10-dimensional representation of the input images
    so that the distances between them minimize the SimNet loss
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # self.conv1 = Conv2D(kernel_size=(3, 3), filters=1, padding='Same',
        #                     activation='relu', input_shape=(28, 28, 1),
        #                     kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.flatten = Flatten()
        # self.dense1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense2 = Dense(10, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    @tf.function
    def call_encoder(self, x: np.array) -> np.array:
        """ Forward pass for one image """
        # x = self.conv1(x)
        x = self.flatten(x)
        # x = self.dense1(x)
        x = self.dense2(x)
        return x

    @tf.function
    def call(self, pair: np.array) -> float:
        """ Forward pass for a pair of images """
        x1, x2 = pair[:, 0, :, :, :], pair[:, 1, :, :, :]
        x1 = self.call_encoder(x1)
        x2 = self.call_encoder(x2)
        difference = x1 - x2
        return difference

    def make_predict(self, pair, threshold=0.5):
        """ pair must have a shape of
        [batch_size (any), 2, 28, 28, 1]
        """
        out = self.call(pair)
        distance_vector = tf.map_fn(lambda x: tf.nn.sigmoid(tf.reduce_sum(tf.square(x))), out)

        # apply threshold
        distance_vector = tf.map_fn(lambda x: 0 if x <= threshold else 1, distance_vector)
        return distance_vector


@tf.function
def simnet_loss(target, difference):
    batch_size = difference.shape[0]
    total_loss = 0

    for i in range(batch_size):
        distance = tf.nn.sigmoid(tf.reduce_sum(tf.square(difference[i, :])))
        loss = (1.0 - target) * tf.square(distance) / 2.0 + target * tf.square(tf.maximum(0.0, 1.0 - distance)) / 2.0
        total_loss += loss

    average_loss = total_loss / batch_size
    print(f'Loss: {average_loss}')
    return average_loss


def load_and_split():
    # load the data
    x, y = get_mnist()
    # select a pair policy
    pairs = RandomSelectionPolicy(random_state=42).select_pairs(100, x, y)
    x, y = Ordering.get_consecutive_pairs(pairs)
    x = np.expand_dims(x, axis=4)
    return train_test_split(x, y, test_size=0.25)


def set_callbacks(model_path=None) -> list:
    # define callbacks
    cbcks = []
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  verbose=1, patience=5, min_lr=0.0001)
    cbcks.append(reduce_lr)

    if model_path:
        mcp_save = ModelCheckpoint(model_path,  # {epoch:02d}-{val_loss:.2f}.hdf5
                                   save_best_only=True, monitor='val_loss', mode='min')
        cbcks.append(mcp_save)
    return cbcks


def train(model: tf.keras.Model, x_train, x_test, y_train, y_test,
          model_path: str = None, n_epoch: int = 10, batch_size: int = 2):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer,
                  loss=simnet_loss)

    # fit and check validation data
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=n_epoch, workers=8,
                        callbacks=set_callbacks(model_path),
                        validation_data=(x_test, y_test))

    # model.save('./benchmarks/encoder_' + str(np.around(history.history['val_loss'][-1], 3)))
    model.summary()

    # tf.keras.utils.plot_model(model, 'simnet_model.png', show_shapes=True, expand_nested=True)
    return history, model


if __name__ == '__main__':

    from sklearn.preprocessing import StandardScaler

    model = Encoder()
    x_train, x_test, y_train, y_test = load_and_split()

    # normalize
    n = x_train.shape[0] + x_test.shape[0]
    X = np.concatenate([x_train, x_test]).reshape((n, -1))
    X_scaled = StandardScaler().fit_transform(X).reshape((n, 2, 28, 28, 1))
    x_train, x_test = X_scaled[:x_train.shape[0]], X_scaled[x_train.shape[0]:]

    # train model
    history, trained_model = train(model, x_train, x_test, y_train, y_test, n_epoch=20, batch_size=1)

    # calculate custom predict function for the test set
    out = trained_model.make_predict(x_test, threshold=0.5)
    accuracy = tf.keras.metrics.binary_accuracy(out, np.array(y_test))
    print('Accuracy on the test set:')
    tf.print(accuracy)

    plot_loss(history)
