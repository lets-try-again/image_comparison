import numpy as np
import matplotlib.pyplot as plt


def plot_loss(history):

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # ax1.plot(history.history['accuracy'], label='accuracy')
    # ax1.plot(history.history['val_accuracy'], label='val_accuracy', dashes=[6, 2])
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Accuracy')
    # ax1.legend(loc='lower right')

    fig, ax2 = plt.subplots(1, 1)

    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss', dashes=[6, 2])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='top right')

    plt.savefig('result.png')
    plt.show()


def plot_pair(pair, y=None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(f'A pair of images with y = {int(y)}')
    ax1.imshow(np.squeeze(pair[0]), cmap='gray')
    ax2.imshow(np.squeeze(pair[1]), cmap='gray')

    plt.show()
