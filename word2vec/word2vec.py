from __future__ import division
from __future__ import print_function

import math
import os
import errno
import sys
import random
from tempfile import gettempdir
import zipfile

import argparse
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from datetime import datetime

from data_utils import (
    build_dataset,
    get_precomputed_dataset,
    generate_batch
)

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate word embedding from a corpus.')

    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        help="The file where the resulting embedding will be saved",
        default="data.npy"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="The file containing a corpus",
    )
    parser.add_argument(
        "output_dict",
        type=str,
        nargs="?",
        help="The file where the dict will be saved",
        default="dict.txt"
    )
    parser.add_argument(
        "-vs",
        "--vocabulary_size",
        type=int,
        nargs="?",
        help="How many words will be \"known\" to the model",
        default=50000
    )
    parser.add_argument(
        "-es",
        "--embedding_size",
        type=int,
        nargs="?",
        help="How many features will be used to describe a word (vector length)",
        default=128
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=128
    )
    parser.add_argument(
        "-sw",
        "--skip_window",
        type=int,
        nargs="?",
        help="How many numbers to consider left and right",
        default=1
    )
    parser.add_argument(
        "-ns",
        "--num_skips",
        type=int,
        nargs="?",
        help="How many times to reuse an input to generate a label",
        default=2
    )
    parser.add_argument(
        "-nsa",
        "--num_sampled",
        type=int,
        nargs="?",
        help="Number of negative examples to sample",
        default=64
    )
    parser.add_argument(
        "-pd",
        "--precomputed_data_files",
        nargs="+",
        help="A list of .npy files containing the dataset (if it was precomputed)"
    )
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        nargs="?",
        help="The folder were the training loss will be recorded. Will be created if not pre-existing"
    )


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

def main():
    """
        Create the embedding using a corpus passed as an argument
    """

    args = parse_arguments()

    vocabulary_size = args.vocabulary_size
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    skip_window = args.skip_window
    num_skips = args.num_skips
    num_sampled = args.num_sampled
    precomputed_data_files = args.precomputed_data_files
    log_dir = args.log_dir

    if log_dir:
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        log_dir = '{}/run_{}/'.format(log_dir, now)
        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    data = None
    data_list = None

    if precomputed_data_files is None or len(precomputed_data_files) == 0:
        print('No precomputed data given!')
        data, data_list = build_dataset(args.input_file, args.output_dict, vocabulary_size)
    else:
        print('Loading .npy files')
        data, data_list = get_precomputed_dataset(precomputed_data_files)

    data_index = 0
    data_list_index = 0

    graph = tf.Graph()

    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device('/cpu:0'):
          embeddings = tf.Variable(
              tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
          embed = tf.nn.embedding_lookup(embeddings, train_inputs)

          nce_weights = tf.Variable(
              tf.truncated_normal([vocabulary_size, embedding_size],
                                  stddev=1.0 / math.sqrt(embedding_size)))
          nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        normalized_embeddings = embeddings / norm

        init = tf.global_variables_initializer()

        loss_summary = tf.summary.scalar('Loss', loss)

        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    num_steps = 100001

    with tf.Session(graph=graph) as session:
        init.run()

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data, data_list, data_index, data_list_index)

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                file_writer.add_summary(
                    loss_summary.eval(feed_dict=feed_dict),
                    step
                )
                average_loss = 0

        np.save(args.output_file, normalized_embeddings.eval())

if __name__=='__main__':
    main()
