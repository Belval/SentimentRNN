import collections
import numpy as np

def read_chunk(filename):
    """
        Read a file in chunks of 10MB, returns a generator.
    """

    with open(filename, 'r') as f:
        while True:
            data = f.read(10 * 1024 * 1024)
            if not data:
                break
            yield data.split()[:-1]

def build_dataset(filename, n_words):
    """
        Build our dataset in 3 steps
    """

    data = []
    word_dict = {}
    dictionary = {'UNK':0,}
    i = 0
    # First reading
    print('Building dictionary')
    for chunk in read_chunk(filename):
        if i % 10 == 0:
            print(str(i) + ' chunk processed')
        for word, c in collections.Counter(chunk).most_common(n_words - 1):
            if not word in word_dict:
                word_dict[word] = c
            else:
                word_dict[word] += c
        i += 1

    for k in sorted(word_dict, key=word_dict.get, reverse=True):
        if len(dictionary) < n_words:
            dictionary[k] = len(dictionary)
        else:
            break

    i = 0
    # Second reading
    print('Building dataset')
    for chunk in read_chunk(filename):
        if i % 10 == 0:
            print(str(i) + ' chunk processed')
        for word in chunk:
            index = dictionary.get(word, 0)
            data.append(index)
        i += 1

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reversed_dictionary

def generate_batch(batch_size, num_skips, skip_window):
    """
        Create a batch
    """

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
