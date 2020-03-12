from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense


class Encoder(Model):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(kernel_size=(3, 3), filters=12, padding = 'Same',
                            activation ='relu', input_shape = (28, 28, 1))
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


if __name__ == '__main__':

    from preprocessing.loader import get_mnist, Ordering
    from preprocessing.pairselector import RandomSelectionPolicy

    x_train, y_train, x_test, y_test = get_mnist()

    pairs = RandomSelectionPolicy().select_pairs(1, x_train, y_train)
    x_train, y_train = Ordering.get_consecutive_pairs(pairs)

    model = Encoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def loss(x1, x2, target):
        def D(t1, t2):
            return np.linalg.norm(t1 - t2)

        return (1 - target) * tf.square(D(x1, x2)) / 2 + \
               target * tf.square(tf.maximum(0, 1 - D(x1, x2) * D(x1, x2))) / 2

    model.compile(optimizer, loss=loss)


    @tf.function
    def train_step(x_train, y_train):

        with tf.GradientTape() as tape:
            for X, y in zip(x_train, y_train):
                x1, x2 = X[0], X[1]
                gradients = tape.gradient(loss(x1, x2, y), model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# =====================================

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
