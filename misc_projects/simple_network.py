"""
Implementation of simple neural network from Week 1 lectures
"""
import os
import numpy as np


SMALL_FILE_PATH = '~/Documents/School/Ryerson_program/CMTH642/Assignment2'
BIG_FILE_PATH = '~/Documents/Code/kaggle/handwriting'
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


def analyze_pictures():
    """
    Main wrapper routine
    :return:
    """
    pass


