import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell

from data_manager import DataManager

class SentimentRNN(object):
    def __init__(self, batch_size, embedding_path, wordlist_path, examples_path, model_path, max_length, train_test_ratio):
        self.__data_manager = DataManager(batch_size, embedding_path, wordlist_path, examples_path, max_length, train_test_ratio)
        self.__model_path = model_path
        self.__training_name = str(int(time.time()))

    def sentiment_rnn(self, session, max_length, input_size):
        input_units = tf.placeholder(tf.float32, (None, max_length, input_size))
        sequence_lengths = tf.placeholder(tf.float32, (None, 1))
        output_unit = tf.placeholder(tf.float32, (None, 1))

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

        writer = tf.summary.FileWriter(os.path.join(self.__model_path, self.__training_name), graph=session.graph)

        init = tf.global_variables_initializer()

        return input_units, sequence_lengths, output_unit, loss, init

    def train(self, iteration_count):
        with tf.Session() as sess:
            input_units, sequence_lengths, output_unit, loss, init = self.sentiment_rnn(
                sess,
                self.__data_manager.max_length,
                self.__data_manager.get_input_size()
            )
            init.run()
            print('Training')
            for i in range(iteration_count):
                batch_y, batch_sl, batch_x = self.__data_manager.get_next_train_batch()
                print(batch_x)
                print(batch_sl)
                print(batch_y)
                loss = sess.run(
                    [loss],
                    feed_dict={
                        input_units: batch_x,
                        sequence_lengths: batch_sl,
                        output_unit: batch_y
                    }
                )
        return None

    def test(self):


        return None
