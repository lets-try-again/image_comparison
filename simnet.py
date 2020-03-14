from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from time import sleep


class Encoder(Model):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(kernel_size=(3, 3), filters=1, padding='Same',
                            activation='relu', input_shape=(28, 28, 1),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense2 = Dense(10, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    @tf.function
    def call_encoder(self, x: np.array) -> np.array:
        """ Forward pass for one image """

        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    @staticmethod
    @tf.function
    def distance(x1: np.array, x2: np.array):
        """ Defines a distance between encoded vectors """
        print('Calculating distance')
        d = (x1 - x2)
        print(f'Shape of d: {d.shape}')
        return d

    @tf.function
    def call(self, pair: np.array) -> float:
        """ Forward pass for a pair of images """

        print(f'Method call() obtained a pair of shape {pair.shape}')
        x1, x2 = pair[:, 0, :, :], pair[:, 1, :, :]

        print(f'Processing x1, x2 with shapes {x1.shape}, {x2.shape}')
        x1 = self.call_encoder(x1)
        x2 = self.call_encoder(x2)

        d = Encoder.distance(x1, x2)
        return d


@tf.function
def simnet_loss(distance, target):

    distance = tf.norm(distance)
    loss = (1.0 - target) * tf.square(distance) / 2.0 + \
           target * tf.square(tf.maximum(0.0, 1.0 - distance * distance)) / 2.0

    print(f'Loss: {loss}')
    return loss


if __name__ == '__main__':

    from preprocessing.loader import get_mnist, Ordering
    from preprocessing.pairselector import RandomSelectionPolicy
    from sklearn.model_selection import train_test_split

    from utils.plot_loss import plot_loss

    tf.config.experimental.set_visible_devices([], 'GPU')

    # load the data
    x, y = get_mnist()
    # select a pair policy
    pairs = RandomSelectionPolicy(random_state=42).select_pairs(100, x, y)
    x, y = Ordering.get_consecutive_pairs(pairs)
    x = np.expand_dims(x, axis=4)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # create a model
    model = Encoder()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    model.compile(optimizer=optimizer,
                  loss=simnet_loss,
                  metrics=['accuracy'])

    # fit and check validation data
    history = model.fit(x_train, y_train,
                        batch_size=2, epochs=120, workers=8,
                        validation_data=(x_test, y_test))

    model.summary()
    # tf.keras.utils.plot_model(model, 'simnet.png', show_shapes=True)
    plot_loss(history)
