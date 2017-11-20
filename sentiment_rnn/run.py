import os
import sys
import argparse

from sentiment_rnn import SentimentRNN


def parse_arguments():
    """
        Parse the commandline arguments
    """

    parser = argparse.ArgumentParser(description='Train and use a RNN for sentiment analysis.')

    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we test the model"
    )
    parser.add_argument(
        "-ttr",
        "--train_test_ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=0.70
    )
    parser.add_argument(
        "-em",
        "--embedding_path",
        type=str,
        nargs="?",
        help="The path where the embedding to use is.",
        required=True
    )
    parser.add_argument(
        "-wl",
        "--wordlist_path",
        type=str,
        nargs="?",
        help="The path with the wordlist matching the embedding is",
        required=True
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        required=True
    )
    parser.add_argument(
        "-ex",
        "--examples_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        required=True
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=64
    )
    parser.add_argument(
        "-it",
        "--iteration_count",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=50000
    )
    parser.add_argument(
        "-ml",
        "--max_length",
        type=int,
        nargs="?",
        help="Maximum sequence length before truncating in words",
        default=200
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

def main():
    """
        Entry point when using SentimentRNN from the commandline
    """

    args = parse_arguments()

    if not args.train and not args.test:
        print("If we are not training, and not testing, what us the point?")

    sentiment_rnn = None

    if args.train:
        sentiment_rnn = SentimentRNN(
            args.batch_size,
            args.embedding_path,
            args.wordlist_path,
            args.examples_path,
            args.model_path,
            args.max_length,
            args.train_test_ratio
        )

        sentiment_rnn.train(args.iteration_count)
        sentiment_rnn.save()

    if args.test:
        if sentiment_rnn is None:
            sentiment_rnn = SentimentRNN(
                args.batch_size,
                args.embedding_path,
                args.wordlist_path,
                args.examples_path,
                args.model_path,
                args.max_length,
                1
            )

        sentiment_rnn.test()

if __name__=='__main__':
    main()
