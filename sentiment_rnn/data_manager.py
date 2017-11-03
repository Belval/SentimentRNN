class DataManager(object):
    def __init__(train_test_ratio, batch_size, embedding_path, data_path):
        if train_test_ratio > 1.0 or train_test_ratio < 0:
            raise Exception('Incoherent ratio!')

        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.current_train_offset = 0
        self.current_test_offset = 0
        self.test_offset = -1
        self.embedding_path = embedding_path
        self.data_path = data_path
        self.data, self.data_len = self.__load_data()

    def __load_data(self):

    def convert_to_vector

    def get_next_train_batch():

        old_offset = self.current_train_offset

        new_offset = self.current_train_offset + self.batch_size

        if new_offset > self.test_offset:
            return []

        self.current_train_offset = new_offset

        return self.data[old_offset:new_offset]

    def get_next_test_batch():
        old_offset = self.current_test_offset

        new_offset = self.current_test_offset + self.batch_size

        if new_offset > self.data_len:
            return []

        self.current_test_offset = new_offset

        return self.data[old_offset:new_offset]
