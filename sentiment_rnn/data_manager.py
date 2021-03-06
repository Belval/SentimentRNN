import numpy as np
from nltk.tokenize import word_tokenize

class DataManager(object):
    def __init__(self, batch_size, embedding_path, wordlist_path, examples_path, max_length, train_test_ratio):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        if max_length < 1:
            raise Exception('Max length should be above 0')

        self.train_test_ratio = train_test_ratio
        self.max_length = max_length
        self.batch_size = batch_size
        self.current_train_offset = 0
        self.embedding = np.load(embedding_path)
        self.examples_path = examples_path
        self.wordlist_path = wordlist_path
        self.wordlist = self.__load_wordlist(wordlist_path)
        self.data, self.data_len = self.__load_data()
        self.test_offset = int(train_test_ratio * self.data_len)
        self.current_test_offset = self.test_offset

    def __load_wordlist(self, wordlist_path):
        with open(wordlist_path, 'r') as wlf:
            return {word.replace('\n', '').lower() : i for i, word in enumerate(wlf.readlines())}

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
        vecs = np.zeros((self.max_length, self.get_input_size()))

        seq_len = 0

        for i, word in enumerate(word_tokenize(sentence)):
            try:
                # We know that word!
                vecs[i, :] = self.embedding[self.wordlist[word.lower()], :]
            except Exception as e:
                # We don't know that word!
                vecs[i, :] = self.embedding[0, :]
            seq_len = i

        return vecs, seq_len

    def get_input_size(self):
        return len(self.embedding[0, :])

    def get_next_train_batch(self):
        while not self.current_train_offset + self.batch_size > self.test_offset:
            old_offset = self.current_train_offset

            new_offset = self.current_train_offset + self.batch_size

            self.current_train_offset = new_offset

            raw_batch_y, raw_batch_x, raw_batch_sl = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_sl = np.reshape(
                np.array(raw_batch_sl),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_length, self.get_input_size())
            )

            yield batch_y, batch_sl, batch_x

        self.current_train_offset = 0

    def get_next_test_batch(self):
        while not self.current_test_offset + self.batch_size > self.data_len:
            old_offset = self.current_test_offset

            new_offset = self.current_test_offset + self.batch_size

            self.current_test_offset = new_offset

            raw_batch_y, raw_batch_x, raw_batch_sl = zip(*self.data[old_offset:new_offset])

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_sl = np.reshape(
                np.array(raw_batch_sl),
                (-1)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_length, self.get_input_size())
            )

            yield batch_y, batch_sl, batch_x
