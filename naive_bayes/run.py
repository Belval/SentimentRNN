"""
    This is meant as a baseline to quantify how much better our model is agains't a bayesian model.
"""

import argparse

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB

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

    return parser.parse_args()

def load_data(path):
    reviews = []
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            try:
                split_line = l.split(';')
                reviews.append((';'.join(split_line[1:]), 'positive' if int(l[0]) >= 3 else 'negative'))
            except Exception as e:
                print(e)
                # Line was malformed
                pass
    return reviews

def main():
    print("Parsing args")
    args = parse_arguments()

    print("Loading reviews")
    reviews = load_data(args.data_path)
    print("{} reviews loaded".format(len(reviews)))
    reviews_features = []

    print("Building word dict")
    words = nltk.FreqDist(
        [
            w.lower()
            for r in list(zip(*reviews))[0] for w in word_tokenize(r)
            if w not in stopwords.words("english") and w not in stopwords.words("french")
        ]
    )

    selected_words = list(words.keys())[0:args.word_count]

    # We will only keep the first X words
    word_features = dict([(w, i) for i, w in enumerate(selected_words)])

    print("Parsing reviews")
    for review, emotion in reviews:
        word_vec = np.zeros(args.word_count)
        for w in word_tokenize(review):
            try:
                word_vec[word_features[w]] = 1
            except Exception as e:
                pass
        reviews_features.append((word_vec, emotion))

    print("Training Bayes classifier")
    train_data, train_gt = zip(*reviews_features[0:int(len(reviews_features) * args.train_test_ratio)])
    test_data, test_gt = zip(*reviews_features[int(len(reviews_features) * args.train_test_ratio):])

    naive_bayes = GaussianNB()

    classifier = naive_bayes.fit(train_data, train_gt)

    print("Testing results")
    train_pred = classifier.predict(train_data)
    test_pred = classifier.predict(test_data)

    good_in_train = (train_pred == train_gt).sum()
    good_in_test = (test_pred == test_gt).sum()

    print("Score on training set: {}/{} ({} %)".format(good_in_train, len(train_data), int(good_in_train / len(train_data) * 100)))
    print("Score on test set: {}/{} ({} %)".format(good_in_test, len(test_data), int(good_in_test / len(test_data) * 100)))

if __name__=='__main__':
    main()
