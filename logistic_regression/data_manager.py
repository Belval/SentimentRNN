import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataManager(object):
    def __init__(self, batch_size, word_count, examples_path, train_test_ratio):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.word_count = word_count
        self.current_train_offset = 0
        self.examples_path = examples_path
        self.word_dict = self.__create_word_dict()
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset

    def __create_word_dict(self):
        print("Building stopword list")
        stopwords_list = set(stopwords.words("english") + stopwords.words("french"))

        print("Building word dict")
        words = nltk.FreqDist(
            [
                w.lower()
                for r in open(self.examples_path, 'r') for w in word_tokenize(r[2:]) # First 2 are the rating!
                if w not in stopwords_list
            ]
        )

        selected_words = list(words.keys())[0:args.word_count]

        # We will only keep the first X words
        return dict([(w, i) for i, w in enumerate(selected_words)])

    def __load_data(self):
        print('Loading data')

        data = []

        with open(self.examples_path, 'r') as df:
            count = 0
            for line in df:
                count += 1
                token_index = line.index(';')
                data.append((float(line[0:token_index]) * 0.20, *self.convert_to_vector(line[token_index+1:])))
                if count % 1000 == 0:
                    print(count)
                if count > 10000:
                    break

        return (data, len(data))

    def convert_to_vector(self, sentence):
        vec = np.zeros(self.word_count)

        for i, word in enumerate(word_tokenize(sentence)):
            try:
                # We know that word!
                vecs[self.word_dict[word]] = 1
            except Exception as e:
                pass

        return vec

    def get_input_size(self):
        return self.word_count

    def get_next_train_batch(self):
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_y, raw_batch_x = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.get_input_size())
            )

            yield batch_y, batch_x

        self.current_train_offset = 0

    def get_next_test_batch(self):
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_y, raw_batch_x = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.get_input_size())
            )

            yield batch_y, batch_x
