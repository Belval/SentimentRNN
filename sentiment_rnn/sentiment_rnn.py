import numpy as np
import tensorflow as tf

from .data_manager import DataManager

class SentimentRNN(object):
    def __init__(train_test_ratio, data_folder, model_folder):
        self.__data_manager = DataManager(train_test_ratio, data_folder)
        self.__model_folder = model_folder

    def sentiment_rnn(self):
        return

    def train(self, iteration_count):
        return

    def test(self):
        return
