import numpy as np
import tensorflow as tf

from .data_manager import DataManager

class SentimentRNN(object):
    def __init__(train_test_ratio, embedding_path, wordlist_path, model_path):
        self.__data_manager = DataManager(train_test_ratio, embedding_path, wordlist_path)
        self.__model_folder = model_folder

    def sentiment_rnn(self):
        return

    def train(self, iteration_count):
        return

    def test(self):
        return
