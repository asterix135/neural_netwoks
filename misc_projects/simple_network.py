"""
Implementation of simple neural network from Week 1 lectures
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


BIG_FILE_PATH = '/Users/christophergraham/Documents/Code/kaggle/handwriting'
SMALL_TEST = 'small_test.csv'
SMALL_TRAIN = 'small_train.csv'
BIG_TEST = 'test.csv'
BIG_TRAIN = 'train.csv'
SEED = 234513


class SimpleNetwork:
    """
    Class to keep and update weights for a particular digit representation
    """
    def __init__(self, width, length, value):
        """
        Creates class object, and populates grid value with random normal
        variables, mean = 0, sd = 1
        :param width:
        :param length:
        :param value:
        :return:
        """
        global SEED
        self._width = width
        self._length = length
        self._value = value
        np.random.seed(SEED)
        SEED = int(SEED*1.2)
        # This sets values as normal random, mu=0, sigma=1
        # maybe want to change this?
        self._array = np.random.randn(width * length)
        # plot_digit(self._array)

    def update_vals(self, update_array):
        self._array += update_array

    def compute_distance(self, image):
        """
        Computes distance between self and a test image.
        Returns scalar value
        :param image: numpy array
        :return total_dist: scalar
        """
        diffs = image - self._array
        total_dist = np.sqrt(np.sum(diffs**2))
        return total_dist

    def get_value(self):
        return self._value


def load_big_images(size):
    """
    Create 3 numpy arrays: test data (paramaters only)
        train data (one for paramaters, one for values)
    :return:
    """
    if size == 'BIG':
        train_file = BIG_TRAIN
        test_file = BIG_TEST
    else:
        train_file = SMALL_TRAIN
        test_file = SMALL_TEST
    curr_dir = os.getcwd()
    os.chdir(BIG_FILE_PATH)
    train = pd.read_csv(BIG_TRAIN)
    params_test = pd.read_csv(BIG_TEST).as_matrix()
    values_train = train['label'].as_matrix()
    params_train = train.drop('label', axis=1).as_matrix()
    os.chdir(curr_dir)
    return params_train, values_train, params_test


def plot_digit(image):
    """
    Create graphic representation of pixel array
    :param image: as 1-D numpy array on 0-255 scale
    :return:
    """
    display_image = np.copy(image)
    display_image.shape=(28,28)
    plt.imshow(display_image, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()


def learn_one_image(images_learned, image_features, image_value, lam_val):
    """
    Compare one image to learned images, and update values for correct and
    wrongly predicted images
    :param images_learned:
    :param image_features:
    :param image_value:
    :param image_value:
    :return:
    """
    for image in images_learned:
        # compare image
        pass


def learn_images(images_learned, features, values, num_tests, lam_val):
    """
    Go through number of example images specified by count and update
    images learned as follows
    :param images_learned: dictionary?? of learned images
    :param features: numpy matrix of feature values
    :param values: numpy vector of true values
    :param count: scalar - number of images to process
    :param lam_val: scalar - amount to adjust learned image +/-
    :return:
    """
    # TODO: Consider whether to put in a convergence value rather than count
    for i in range(num_tests):
        nearest = float('inf')
        pred = 99
        for image in images_learned:
            img_distance = image.compute_distance(features[i])
            if img_distance < nearest:
                nearest = img_distance
                pred = image.get_value()
        # TODO: need to update dictionary appropriately here



def analyze_pictures():
    """
    Main wrapper routine
    :return:
    """
    params_train, values_train, params_test = load_big_images('SMALL')
    # TODO: Decide if this should be a dictionary or list
    images_learned = {}
    for val in range(10):
        images_learned[val] = SimpleNetwork(28, 28, val)
    print(images_learned[0].compute_distance(params_train[0]))


if __name__ == '__main__':
    analyze_pictures()