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
        self.__session = tf.Session()

        with self.__session.as_default():
            self.__input_units, self.__sequence_lengths, self.__output_unit, self.__loss, self.__init, self.__predicted_output, self.__train = self.sentiment_rnn(
                self.__data_manager.max_length,
                self.__data_manager.get_input_size()
            )
            self.__init.run()

    def sentiment_rnn(self, max_length, input_size):
        input_units = tf.placeholder(tf.float32, (None, max_length, input_size))
        sequence_lengths = tf.placeholder(tf.int32, (self.__data_manager.batch_size))
        output_unit = tf.placeholder(tf.float32, (None))

        gru_cell = GRUCell(input_size)

        init_state = tf.get_variable(
            'init_state',
            [1, input_size],
            initializer=tf.constant_initializer(0.0)
        )

        init_state = tf.tile(init_state, [self.__data_manager.batch_size, 1])

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            gru_cell,
            input_units,
            sequence_length=sequence_lengths,
            initial_state=init_state,
            dtype=tf.float32
        )

        rnn_outputs = tf.nn.dropout(rnn_outputs, tf.constant(1.0))

        idx = sequence_lengths - 1
        last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, input_size]), idx)

        W = tf.Variable(tf.random_uniform([input_size, 1]))

        predicted_output = tf.reshape(tf.matmul(last_rnn_output, W), [self.__data_manager.batch_size], name='predictions')

        loss = tf.reduce_mean(tf.squared_difference(output_unit, predicted_output))

        tf.summary.scalar('Loss', loss)

        train = tf.train.AdamOptimizer().minimize(loss)

        accuracy = tf.reduce_mean(tf.cast(tf.abs(output_unit - predicted_output), tf.float32))

        tf.summary.scalar('Accuracy', accuracy)

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter(os.path.join(self.__model_path, self.__training_name), graph=self.__session.graph)

        init = tf.global_variables_initializer()

        return input_units, sequence_lengths, output_unit, loss, init, predicted_output, train

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(iteration_count):
                iter_loss = 0
                total_error = 0
                example_count = 0
                for batch_y, batch_sl, batch_x in self.__data_manager.get_next_train_batch():
                    _, loss_value, pred_output = self.__session.run(
                        [self.__train, self.__loss, self.__predicted_output],
                        feed_dict={
                            self.__input_units: batch_x,
                            self.__sequence_lengths: batch_sl,
                            self.__output_unit: batch_y
                        }
                    )
                    example_count += len(batch_y)
                    total_error += np.sum(abs(batch_y - pred_output))
                    iter_loss += loss_value
                print('[{}] Iteration loss: {} Iteration total error: {} (about {} per example)'.format(i, iter_loss, total_error, total_error / example_count))
        return None

    def test(self, text=None):
        with self.__session.as_default():
            print('Testing')
            total_error = 0
            example_count = 0
            for batch_y, batch_sl, batch_x in self.__data_manager.get_next_test_batch():
                pred_output = self.__session.run(
                    [self.__predicted_output],
                    feed_dict={
                        self.__input_units: batch_x,
                        self.__sequence_lengths: batch_sl
                    }
                )
                example_count += len(batch_y)
                total_error += np.sum(abs(batch_y - pred_output))
            print('Error on test set: {} (about {} for every example)'.format(total_error, total_error / example_count))
        return None

    def save(self, path=None):
        path = os.path.join(self.__model_path, self.__training_name) if path is None else path

        with open(path + '/sentiment_rnn.pb', 'wb') as f:
            f.write(
                tf.graph_util.convert_variables_to_constants(
                    self.__session,
                    self.__session.graph.as_graph_def(),
                    ['predictions']
                ).SerializeToString()
            )
        return None
