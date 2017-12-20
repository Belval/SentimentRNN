import collections
import uuid
import random
import nltk
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

def build_dataset(filename, dict_filename, n_words):
    """
        Build our dataset in 3 steps
    """

    data = []
    data_list = []
    word_dict = {}
    dictionary = {'UNK':0,}
    chunk_index = 0
    # First reading
    print('Building dictionary')
    for chunk in read_chunk(filename):
        if chunk_index % 10 == 0:
            print(str(chunk_index) + ' chunk processed')
        for word, c in collections.Counter(chunk).most_common(n_words - 1):
            if not word in word_dict:
                word_dict[word] = c
            else:
                word_dict[word] += c
        chunk_index += 1

    for k in sorted(word_dict, key=word_dict.get, reverse=True):
        if len(dictionary) < n_words:
            dictionary[k] = len(dictionary)
        else:
            break

    chunk_index = 0
    wc = 0
    # Each array is a billion unsigned int, if we get bigger than that, we'll save it and create a new one
    current_np_array = np.zeros(1000000000, dtype='uint32')
    # Second reading
    print('Building dataset')
    for chunk in read_chunk(filename):
        if chunk_index % 10 == 0:
            print(str(chunk_index) + ' chunk processed')
        for word in chunk:
            current_np_array[wc % 1000000000] = dictionary.get(word, 0)
            wc += 1
            if wc % 1000000000 == 0:
                data_list.append(str(uuid.uuid4()) + '.npy')
                np.save(data_list[-1], current_np_array)
                current_np_array = np.zeros(1000000000, dtype='uint32')

        chunk_index += 1

    # Saving dictionary (as a list of word)
    with open(dict_filename, 'w') as dict_file:
        for k in sorted(dictionary, key=word_dict.get):
            dict_file.write(k + '\n')

    data = np.load(data_list[0])
    return data, data_list

def get_precomputed_dataset(data_files):
    """
        If the data files are already computed, you can skip processing by calling
        this instead of build_dataset
    """

    data = np.load(data_files[0])
    return data, data_files

def generate_batch(batch_size, num_skips, skip_window, data, data_list, data_index, data_list_index):
    """
        Create a batch
    """

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
        data_list_index = data_list_index + 1 if data_list_index + 1 < len(data_list) else 0
        data = np.load(data_list[data_list_index])
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
