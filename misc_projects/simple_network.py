"""
Implementation of simple neural network from Week 1 lectures
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SMALL_FILE_PATH = '/Users/christophergraham/Documents/School/Ryerson_program/CMTH642/CMTH642_Assignments/Assignment2'
BIG_FILE_PATH = '/Users/christophergraham/Documents/Code/kaggle/handwriting'
SMALL_TEST = 'pendigits.tes.csv'
SMALL_TRAIN = 'pendigits.tra.csv'
BIG_TEST = 'test.csv'
BIG_TRAIN = 'train.csv'


class SimpleNetwork:
    """
    Class to keep and update weights for a particular digit representation
    """
    def __init__(self, width, length, value):
        self._width = width
        self._length = length
        self._value = value
        self._array = np.random.randn(width, length)

    def update_vals(self, update_array):
        self._array += update_array

    def __str__(self):
        pass




def load_big_images():
    pass


def load_small_images():
    pass


def show_digit():
    """
    Create graphic representation of pixel array
    :return:
    """
    pass


def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


def analyze_pictures():
    """
    Main wrapper routine
    :return:
    """
    pass


curr_dir = os.getcwd()
print(curr_dir)
os.chdir(SMALL_FILE_PATH)
# load training data
test_set = pd.read_csv(SMALL_TRAIN, header=None)
#select one image to test print
test_image = test_set.iloc[[0]].drop(16, axis=1).as_matrix()
test_image.shape=(4,4)
print(test_image)
imgplot = plt.imshow(test_image)
print(imgplot)

