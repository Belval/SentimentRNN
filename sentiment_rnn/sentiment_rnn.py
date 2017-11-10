import numpy as np
import tensorflow as tf

from tf.nn.rnn_cell import GRUCell

from .data_manager import DataManager

class SentimentRNN(object):
    def __init__(train_test_ratio, embedding_path, wordlist_path, model_path):
        self.__data_manager = DataManager(train_test_ratio, embedding_path, wordlist_path)
        self.__model_folder = model_folder

    def sentiment_rnn(self, input_size):
        input_units = tf.placeholder(tf.float32, (None, None, input_size))
        output_units = tf.placeholder(tf.float32, (None, None, 1))

        gru_cell = GRUCell(input_size)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(gru_cell, input_units, dtype=tf.float32)

        predicted_output = tf.map_fn(
            lambda x: tf.contrib.layers.linear(x, num_outputs=1, activation_fn=None),
            rnn_outputs
        )

        loss = tf.reduce_mean(tf.squared_difference(output_unit, predicted_output))

        tf.summary.scalar('Loss', loss)

        train = tf.train.AdamOptimizer().minimize(loss)

        accuracy = tf.reduce_mean(tf.cast(tf.abs(output_unit - predicted_output), tf.float32))

        tf.summary.scalar('Accuracy', accuracy)

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(os.path.join(args[1], training_uuid), graph=session.graph)

        return output_units, loss

    def train(self, iteration_count):
        return

    def test(self):
        return
