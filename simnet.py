import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# to run using CPU only
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras import Model
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
        super(Encoder, self).__init__()
        self.cv = Conv2D(12, (3, 3), activation='relu', padding='Same',
                         kernel_initializer='he_uniform', input_shape=(28, 28, 1),
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.pool = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(10, activation=None,
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))

    @tf.function
    def call_encoder(self, x: np.array) -> np.array:
        """ Forward pass for one image """
        x = self.cv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    @tf.function
    def call(self, pair: np.array) -> float:
        """ Forward pass for a pair of images """
        x1, x2 = pair[:, 0, :, :, :], pair[:, 1, :, :, :]
        x1 = self.call_encoder(x1)
        x2 = self.call_encoder(x2)
        difference = x1 - x2
        return difference

    @staticmethod
    def distance(difference):
        """ The D function from the paper which is used in loss """
        return tf.nn.tanh(tf.reduce_sum(tf.sqrt(tf.square(difference))))

    def make_predict(self, pair, threshold=0.5):
        """ pair must have a shape of
        [batch_size (any), 2, 28, 28, 1]
        """
        out = self.call(pair)
        distance_vector = tf.map_fn(lambda x: Encoder.distance(x), out)
        # apply threshold
        distance_vector = tf.map_fn(lambda x: 0 if x <= threshold else 1, distance_vector)
        return distance_vector


@tf.function
def custom_accuracy(y_true, y_pred):
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), y_pred)
    distance_vector = tf.map_fn(lambda x: 0.0 if x <= 0.5 else 1.0, distance_vector)
    accuracy = tf.keras.metrics.binary_accuracy(y_true, distance_vector)
    return accuracy


@tf.function
def simnet_loss(target, difference):
    distance_vector = tf.map_fn(lambda x: Encoder.distance(x), difference)
    loss = tf.map_fn(lambda distance: target * tf.square(distance) +
                                      (1.0 - target) * tf.square(tf.maximum(0.0, 1.0 - distance)), distance_vector)
    average_loss = tf.reduce_mean(loss)
    return average_loss


def train(model: tf.keras.Model, x_train, x_test, y_train, y_test,
          model_path: str = None, n_epoch: int = 10, batch_size: int = 2):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss=simnet_loss,
                  metrics=[custom_accuracy])

    # fit and check validation data
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=n_epoch, workers=8,
                        # callbacks=set_callbacks(model_path),
                        validation_data=(x_test, y_test))

    # model.save('./benchmarks/encoder_' + str(np.around(history.history['val_loss'][-1], 3)))
    model.summary()

    # tf.keras.utils.plot_model(model, 'simnet_model.png', show_shapes=True, expand_nested=True)
    return history, model


if __name__ == '__main__':

    model = Encoder()
    x_train, x_test, y_train, y_test = load_and_split(n_pairs=2000)
    x_train, x_test = shrink(x_train, x_test)

    # train model
    history, trained_model = train(model, x_train, x_test, y_train, y_test,
                                   n_epoch=2, batch_size=64)

    plot_loss(history)

    # # calculate custom predict function for the test set
    # out = trained_model.make_predict(x_test, threshold=0.5)
    # print(f'Prediction: {np.array(out)}')
    # accuracy = tf.keras.metrics.binary_accuracy(out, np.array(y_test))
    # print('Accuracy on the test set:')
    # tf.print(accuracy)

    # # print weights
    # for layer in trained_model.layers:
    #     weights = layer.get_weights()
    #     print(weights)
    #     for w in weights:
    #         print(f'Sum of weights: {w.sum()}')

    # plot embedding of encoded images
    # images = x_train.reshape((x_train.shape[0] * x_train.shape[1], 28, 28, 1))
    # embeddings_of_test = trained_model.call_encoder(images)
    # points = reduce_dim(np.array(embeddings_of_test))
    # plot_embedding(points)
    # plt.show()

    # # calculate custom predict function for the train set
    # out_train = trained_model.make_predict(x_train, threshold=0.5)
    # accuracy = tf.keras.metrics.binary_accuracy(out_train, np.array(y_train))
    # print('Accuracy on the train set:')
    # tf.print(accuracy)

    # # plot pair
    # # these are y == 0 (different)
    # plot_pair(x_train[-1], y=y_train[-1])
    # # these are y == 1 (same)
    # plot_pair(x_train[0], y=y_train[0])

    # def set_callbacks(model_path=None) -> list:
    #     # define callbacks
    #     cbcks = []
    #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                   verbose=1, patience=5, min_lr=0.01)
    #     cbcks.append(reduce_lr)
    #
    #     if model_path:
    #         mcp_save = ModelCheckpoint(model_path,  # {epoch:02d}-{val_loss:.2f}.hdf5
    #                                    save_best_only=True, monitor='val_loss', mode='min')
    #         cbcks.append(mcp_save)
    #     return cbcks

