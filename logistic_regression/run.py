"""
    This is meant as a baseline to quantify how much better our model is agains't a bayesian model.
"""

import argparse

from logistic_regression import LogisticRegression

def parse_arguments():
    """
        Parse the commandline arguments
    """

    parser = argparse.ArgumentParser(description='Run the basic NLTK text classification.')

    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples.",
        default="../data/save.txt"
    )
    parser.add_argument(
        "-c",
        "--word_count",
        type=int,
        nargs="?",
        help="How many words will be kept as features",
        default=5000
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
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="How many examples will be used per batch",
        default=64
    )
    parser.add_argument(
        "-it",
        "--iteration_count",
        type=int,
        nargs="?",
        help="How many iterations will be done"
    )
    return parser.parse_args()

def main():
    print("Parsing args")
    args = parse_arguments()

    log_reg = LogisticRegression(args.batch_size, args.word_count, args.data_path, args.train_test_ratio)
    log_reg.train(args.iteration_count)

if __name__=='__main__':
    main()
