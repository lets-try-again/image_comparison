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

    @tf.function
    def call(self, pair: np.array) -> float:
        """ Forward pass for a pair of images """

        x1, x2 = pair[:, 0, :, :], pair[:, 1, :, :]
        x1 = self.call_encoder(x1)
        x2 = self.call_encoder(x2)

        return x1 - x2


@tf.function
def simnet_loss(difference, target):

    distance = tf.norm(difference)
    loss = (1.0 - target) * tf.square(distance) / 2.0 + \
           target * tf.square(tf.maximum(0.0, 1.0 - distance * distance)) / 2.0

    print(f'Loss: {loss}')
    return loss


def load_and_split():
    # load the data
    x, y = get_mnist()
    # select a pair policy
    pairs = RandomSelectionPolicy(random_state=42).select_pairs(100, x, y)
    x, y = Ordering.get_consecutive_pairs(pairs)
    x = np.expand_dims(x, axis=4)
    return train_test_split(x, y, test_size=0.25)


def train(model: tf.keras.Model, models_path):

    x_train, x_test, y_train, y_test = load_and_split()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer,
                  loss=simnet_loss)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  verbose=1, patience=5, min_lr=0.0001)

    checkpoints_path = models_path + 'encoder.h5'
    mcp_save = ModelCheckpoint(checkpoints_path,  # {epoch:02d}-{val_loss:.2f}.hdf5
                               save_best_only=True, monitor='val_loss', mode='min')

    # fit and check validation data
    history = model.fit(x_train, y_train,
                        batch_size=2, epochs=2, workers=8,
                        callbacks=[reduce_lr, mcp_save],
                        validation_data=(x_test, y_test))

    # model.save('./benchmarks/encoder_' + str(np.around(history.history['val_loss'][-1], 3)))
    model.summary()
    plot_loss(history)

    # tf.keras.utils.plot_model(model, 'simnet.png', show_shapes=True)
    return checkpoints_path


if __name__ == '__main__':

    models_path = './benchmarks/'

    model = Encoder()
    checkpoint_path = train(model, models_path=models_path)

