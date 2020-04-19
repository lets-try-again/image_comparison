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

from preprocessing.scaler import shrink
from preprocessing.loader import load_and_split
from preprocessing.dimreduct import reduce_dim
from utils.plot_utils import plot_loss, plot_embedding, plot_pair

np.set_printoptions(suppress=True)


class Encoder(Model):
    """
    A network that finds a 10-dimensional representation of the input images
    so that the distances between them minimize the SimNet loss
    """

    def __init__(self):
        super().__init__()

        self.cv = Conv2D(24, (3, 3), activation='relu', padding='Same',
                         input_shape=(28, 28, 1), kernel_initializer=tf.keras.initializers.glorot_normal(),
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.pool = MaxPooling2D((2, 2))
        self.cv2 = Conv2D(24, (3, 3), activation='relu', padding='Same',
                          input_shape=(28, 28, 1), kernel_initializer=tf.keras.initializers.glorot_normal(),
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(5, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(),
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

    @tf.function
    def call_encoder(self, x):
        """ Forward pass for one image """
        x = self.cv(x)
        x = self.pool(x)
        x = self.cv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    @tf.function
    def call(self, pair):
        """ Forward pass for a pair of images """
        x1, x2 = pair[:, 0, :, :, :], pair[:, 1, :, :, :]
        x1 = self.call_encoder(x1)
        x2 = self.call_encoder(x2)
        difference = x1 - x2
        return difference

    @staticmethod
    def distance(difference):
        """ The D function from the paper which is used in loss """
        # return tf.nn.tanh(tf.reduce_mean(tf.sqrt(tf.square(difference))))
        distance = tf.sqrt(tf.reduce_sum(tf.pow(difference, 2), 0))
        # return tf.nn.tanh(tf.norm(difference))
        return distance

    def make_predict(self, pair, threshold=0.5):
        """ pair must have a shape of
        [batch_size (any), 2, 28, 28, 1]
        """
        out = self.call(pair)
        distance_vector = tf.map_fn(lambda x: Encoder.distance(x), out)
        # apply threshold
        distance_vector = tf.map_fn(lambda x: 0.0 if x <= threshold else 1.0, distance_vector)
        return distance_vector


@tf.function
def custom_accuracy(y_true, y_pred):
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), y_pred)
    # distance_vector = tf.map_fn(lambda x: 0.0 if x <= 0.5 else 1.0, distance_vector)
    accuracy = tf.keras.metrics.binary_accuracy(y_true, distance_vector)
    return accuracy


@tf.function
def simnet_loss(target, difference):
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), difference)
    loss = tf.map_fn(lambda distance: target * tf.square(distance) +
                                      (1.0 - target) * tf.square(tf.maximum(0.0, 1.0 - distance)), distance_vector)
    average_loss = tf.reduce_mean(loss)
    return average_loss


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = simnet_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    tf.print(f"Loss:", loss, output_stream=sys.stdout)


@tf.function
def test_step(x_test, y_test, label='test'):
    out = model.make_predict(x_test, threshold=0.5)
    accuracy = tf.keras.metrics.binary_accuracy(out, y_test)
    tf.print(f"Accuracy on the {label} set:", accuracy, output_stream=sys.stdout)


if __name__ == '__main__':

    n_total_pairs = 8000
    n_epoch = 10

    n_pairs = int(np.sqrt(n_total_pairs / 2))
    # print(f'Calling pair selector with n = {n_pairs}')
    x_train, x_test, y_train, y_test = load_and_split(n_pairs=n_pairs, n_classes=2)
    x_train, x_test = shrink(x_train, x_test)
    y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    # print('Percentage of similar images', np.round(y_train.sum()/y_train.shape[0], 3))

    model = Encoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.build(input_shape=(None, 2, 28, 28, 1))
    model.summary()

    for epoch in range(n_epoch):
        print(f'epoch {epoch + 1}')
        train_step(x_train, y_train)
        test_step(x_train, y_train, label='train')
        test_step(x_test, y_test)

    # plot embedding of encoded images
    images = x_train.reshape((x_train.shape[0] * x_train.shape[1], 28, 28, 1))
    embeddings_of_train = model.call_encoder(images)
    if embeddings_of_train.shape[1] > 2:
        print('Applying t-SNE')
        embeddings_of_test = reduce_dim(np.array(embeddings_of_train))
    plot_embedding(embeddings_of_train)
    plt.show()

    ## saving weights
    # model = ResNet34()
    # model.build((1, 224, 224, 3))
    # model.summary()
    # model.save_weights('model_weights.h5')
    #
    # ## loading saved weights
    # model_new = ResNet34()
    # model_new.build((1, 224, 224, 3))
    # model_new.load_weights('model_weights.h5')
