import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# to run using CPU only
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model

from preprocessing.scaler import shrink
from preprocessing.loader import load_and_split
from preprocessing.dimreduct import reduce_dim
from utils.plot_utils import plot_loss, plot_embedding, plot_pair


np.set_printoptions(suppress=True)


class Encoder(Model):
    """
    A network that finds a 50-dimensional representation of the input images
    so that the distances between them minimize the constructive loss
    """

    def __init__(self):
        super(Encoder, self).__init__(name='encoder')

        self.cv = Conv2D(32, (3, 3), activation='relu', padding='Same',
                         input_shape=(28, 28, 1), kernel_initializer=tf.keras.initializers.glorot_normal(),
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.pool = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(50, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(),
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs, training=None, mask=None):
        """ Forward pass for one image """
        x = self.cv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    @staticmethod
    def distance(difference):
        """ The D function from the paper which is used in loss """
        distance = tf.sqrt(tf.reduce_sum(tf.pow(difference, 2), 0))
        return distance

    def make_predict(self, pair, threshold=0.5):
        """ pair must have a shape of
        [batch_size (any), 2, 28, 28, 1]
        """
        x1, x2 = pair[:, 0, :, :, :], pair[:, 1, :, :, :]
        x1 = self.call(x1)
        x2 = self.call(x2)
        difference_vector = x2 - x1
        distance_vector = tf.map_fn(lambda x: Encoder.distance(x), difference_vector)
        # apply threshold
        distance_vector = tf.map_fn(lambda x: 0.0 if x <= threshold else 1.0, distance_vector)
        return distance_vector


def custom_accuracy(y_true, y_pred):
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), y_pred)
    accuracy = tf.keras.metrics.binary_accuracy(y_true, distance_vector)
    return accuracy


def simnet_loss(target, x1, x2):
    difference = x1 - x2
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), difference)
    loss = tf.map_fn(lambda distance: target * tf.square(distance) +
                                      (1.0 - target) * tf.square(tf.maximum(0.0, 1.0 - distance)), distance_vector)
    average_loss = tf.reduce_mean(loss)
    return average_loss


def train_step(images, labels):
    with tf.GradientTape() as tape:
        x1, x2 = images[:, 0, :, :, :], images[:, 1, :, :, :]
        x1 = model(x1)
        x2 = model(x2)
        loss = simnet_loss(labels, x1, x2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test_step(x_test, y_test):
    out = model.make_predict(x_test, threshold=0.5)
    accuracy = tf.keras.metrics.binary_accuracy(out, y_test)
    return accuracy


if __name__ == '__main__':

    n_total_pairs = 10000
    n_epoch = 10
    batch_size = 180
    learning_rate = 0.01

    n_pairs = int(np.sqrt(n_total_pairs / 2))
    print(f'Calling pair selector with n = {n_pairs}')
    x_train, x_test, y_train, y_test = load_and_split(n_pairs=n_pairs, n_classes=2)
    x_train, x_test = shrink(x_train, x_test)
    y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    print('Percentage of similar images', np.round(y_train.sum()/y_train.shape[0], 3))

    model = Encoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(n_epoch):
        epoch_loss = 0
        n_batches = int(x_train.shape[0]/batch_size)
        for indices in np.array_split(np.arange(x_train.shape[0]), indices_or_sections=n_batches):
            x = np.take(x_train, indices, axis=0)
            y = np.take(y_train, indices, axis=0)
            epoch_loss += train_step(x, y)

        epoch_loss = epoch_loss / n_batches
        accuracy = test_step(x_train, y_train)
        val_accuracy = test_step(x_test, y_test)
        tf.print("epoch:", epoch, "loss:", epoch_loss, "accuracy:", accuracy,
                 "val_accuracy:", val_accuracy, output_stream=sys.stdout)

    # plot embedding of encoded images
    # images = x_train.reshape((x_train.shape[0] * x_train.shape[1], 28, 28, 1))
    # embeddings_of_train = model(images)
    # if embeddings_of_train.shape[1] > 2:
    #     print('Applying t-SNE')
    #     embeddings_of_test = reduce_dim(np.array(embeddings_of_train))
    # plot_embedding(embeddings_of_train)
    # plt.show()

    # for i in range(x_train.shape[0]):
    #     plot_pair(x_train[i], y_train[i])
    #     plt.show()

    # model.build(input_shape=(None, 28, 28, 1))
