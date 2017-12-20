import tensorflow as tf
import numpy as np

from data_manager import DataManager

class LogisticRegression(object):
    def __init__(self, batch_size, word_count, examples_path, train_test_ratio):
        self.__data_manager = DataManager(batch_size, word_count, examples_path, train_test_ratio)
        self.__model_path = model_path
        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        with self.__session.as_default():
            self.__input_units, self.__output_unit, self.__loss, self.__init, self.__predicted_output, self.__train = self.sentiment_rnn(
                self.__data_manager.vec_len
            )
            self.__init.run()

    def logistic_regression(self, vec_len):
        # Input layer
        x = tf.placeholder(tf.float32, [None, vec_len])
        # Output layer
        y = tf.placeholder(tf.float32, [1])

        # Weights
        W = tf.Variable(tf.zeros([vec_len]))
        # Bias
        b = tf.Variable(tf.zeros([1]))

        # Prediction
        prediction = tf.reshape(tf.matmul(x, W) + b, [self.__data_manager.batch_size], name='prediction')

        # Loss function
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction)))

        # Optimizer
        optimizer = tf.train.MomentumOptimizer(0.001).minimize(loss)

        init = tf.global_variables_initializer()

        return x, y, loss, init, prediction, optimizer

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(iteration_count):
                iter_loss = 0
                total_error = 0
                example_count = 0
                for batch_y, batch_x in self.__data_manager.get_next_train_batch():
                    _, loss_value, pred_output = self.__session.run(
                        [self.__train, self.__loss, self.__predicted_output],
                        feed_dict = {
                            self.__input_units: batch_x,
                            self.__output_unit: batch_y
                        }
                    )
                    example_count += len(batch_y)
                    total_error += np.sum(abs(batch_y - pred_output))
                    iter_loss += loss_value
                print('[{}] Iteration loss: {} Iteration total error: {} (about {} per example)'.format(i, iter_loss, total_error, total_error / example_count))
        return None

    def test(self):
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

        with open(path + '/logistic_regression.pb', 'wb') as f:
            f.write(
                tf.graph_util.convert_variables_to_constants(
                    self.__session,
                    self.__session.graph.as_graph_def(),
                    ['prediction']
                ).SerializeToString()
            )
        return None
