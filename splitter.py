import os


class Splitter:
    def __init__(self, path):
        self.path = path
        self.dir_ = os.listdir(path)
        # we know it has data split if it contains both test and train files
        self.test_train = list(filter(lambda x: x.endswith("test.conllu") or x.endswith("train.conllu"), self.dir_))
        self.has_train = len(self.test_train) == 2

    def separate_files(self):
        if self.has_train:
            #       train file                                       test file
            return f"{self.path}/{self.test_train[1]}", f"{self.path}/{self.test_train[0]}"
        # otherwise we would need to make own split
        return self._make_split()

    def _make_split(self):
        # test file
        filename = self.test_train[0]
        # divide test file on training set and testing set
        train_file, test_file = filename.replace("test", "train"), filename
        # open files to write and read
        try:
            with open("data_splits/" + train_file, "a", encoding='utf8') as file_writer_train, \
                    open("data_splits/" + test_file, "a", encoding='utf8') as file_writer_test, \
                    open(f"{self.path}/{test_file}", encoding='utf8') as file:
                sent_idx = 0
                # used for clarity, we can do only with one
                is_even, is_odd = False, False
                for line in file:
                    # keep track of sentences
                    if line.startswith("# sent_id"):
                        sent_idx += 1
                        if sent_idx % 2 != 0:
                            is_odd, is_even = True, False
                        else:
                            is_odd, is_even = False, True
                    else:
                        if is_odd:
                            # write odd sentences to test file
                            file_writer_test.write(line)
                        else:
                            # write even sentences to train file
                            file_writer_train.write(line)
        except IOError:
            print("Something went wrong...")
        return f"data_splits/{train_file}", f"data_splits/{test_file}"
