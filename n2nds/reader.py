from n2nds.config import Config
from n2nds.data import Data
import random


class SpToken:
    EOS = "<end>"
    NIL = "<no word>"
    UNK = "<unknown word>"


class WeiboReader:
    def _add_to_vocab(self, word):
        self.vocabulary.setdefault(word, len(self.vocabulary))

    def __init__(self, post_path, response_path, pre_trained_path, batch_size=-1):

        # Load the file
        f_post = open(post_path)
        f_response = open(response_path)
        self._posts = []
        self._responses = []
        for post in f_post.readlines():
            self._posts.append(post.strip('\n'))
        for response in f_response.readlines():
            self._responses.append(response.strip('\n'))

        # Generate the dictionary
        self.vocabulary = None
        self.embedding = None
        if pre_trained_path is None:
            self.vocabulary = {}
            self._add_to_vocab(SpToken.EOS)
            self._add_to_vocab(SpToken.NIL)
            self._add_to_vocab(SpToken.UNK)
            for utters in [self._posts, self._responses]:
                for utter in utters:
                    for ch in utter:
                        self._add_to_vocab(ch)
        else:
            self.vocabulary, self.embedding = EmbeddingReader.load(path=pre_trained_path)

        # Generate the data set
        self._indices, self._lengths, self._weights = \
            self._gen_length_and_weights(zip(self._posts, self._responses))

        # Generate the config
        self.config = Config()
        self.config.BATCH_SIZE = len(self._indices) if batch_size == -1 else batch_size
        self.config.SEQ_SIZE = len(self._indices[0][0])
        self.config.VOCAB_SIZE = len(self.vocabulary)
        self.config.EMBED_SIZE = 20
        self.config.UNIT_SIZE = 20

        # Some variable for batch generation
        self._batch_pointer = 0
        self.dataset_size = len(self._indices)
        self._batch_size = batch_size

        if self.dataset_size % self._batch_size:
            print("data set size is %d, batch size is %d, batch size must be divided by total size" % (
                self.dataset_size, self._batch_size))
            exit(0)

    def next_batch(self):
        next_batch_pointer = min(self._batch_pointer + self._batch_size, self.dataset_size)
        data = Data()
        data.indices = self._indices[self._batch_pointer:next_batch_pointer]
        data.lengths = self._lengths[self._batch_pointer:next_batch_pointer]
        data.weights = self._weights[self._batch_pointer:next_batch_pointer]
        self._batch_pointer = 0 if next_batch_pointer == self.dataset_size else next_batch_pointer
        return data

    def gen_indices_and_lengths(self, sentence):
        data_indices = []
        for word in sentence:
            if word in self.vocabulary:
                data_indices.append(self.vocabulary[word])
            else:
                data_indices.append(self.vocabulary[SpToken.UNK])
        data_indices.append(self.vocabulary[SpToken.EOS])

        sentence_len = len(data_indices)
        data_indices.extend([self.vocabulary[SpToken.NIL]] *
                            (self.config.SEQ_SIZE - sentence_len))
        # Fill several zeros as response and clone sentences into a batch
        data_indices = [[data_indices, [0] * len(data_indices)]]
        data_lengths = [[sentence_len, 0]]
        return data_indices, data_lengths

    def _gen_length_and_weights(self, post_response_pairs):
        post_lengths = []
        response_lengths = []
        post_indices = []
        response_indices = []

        max_length = 0

        # Generate indices from words
        for post, response in post_response_pairs:
            post_index = []
            response_index = []

            for utter_index, utter in zip([post_index, response_index], [post, response]):
                for ch in utter:
                    if ch in self.vocabulary:
                        utter_index.append(self.vocabulary[ch])
                    else:
                        utter_index.append(self.vocabulary[SpToken.UNK])
                utter_index.append(self.vocabulary[SpToken.EOS])

            post_lengths.append(len(post) + 1)
            response_lengths.append(len(response) + 1)

            max_length = max(len(post) + 1, len(response) + 1, max_length)

            post_indices.append(post_index)
            response_indices.append(response_index)

        # Append NIL to the rest of indices
        nil_index = self.vocabulary[SpToken.NIL]
        for post_index in post_indices:
            post_index.extend([nil_index] * (max_length - len(post_index)))
        for response_index in response_indices:
            response_index.extend([nil_index] * (max_length - len(response_index)))

        # Generate lengths
        lengths = list(map(list, zip(post_lengths, response_lengths)))

        # Generate utterances
        utter_indices = list(map(list, zip(post_indices, response_indices)))

        # Generate weights for response
        weights = []
        for response_length in response_lengths:
            weights.append([1] * response_length + [0] * (max_length - response_length))

        return utter_indices, lengths, weights

    def gen_words_from_indices(self, word_indices):
        rev_vocabulary = {v: k for k, v in self.vocabulary.items()}
        return "".join(list(map(rev_vocabulary.get, word_indices)))


class EmbeddingReader:
    @staticmethod
    def load(path):
        def _add_to_vocab(char):
            vocabulary.setdefault(char, len(vocabulary))

        def _add_to_embed(embed):
            embeddings.append(embed)

        def _generate_random(num):
            ret = []
            for i in range(num):
                ret.append(random.uniform(0, 0.01))
            return ret

        vocabulary = dict()
        embeddings = list()

        _add_to_vocab(SpToken.EOS)
        _add_to_vocab(SpToken.NIL)
        _add_to_vocab(SpToken.UNK)
        for i in range(len(vocabulary)):
            _add_to_embed(_generate_random(200))

        f = open(path)

        i = 0
        while True:
            # i += 1
            # print(i)

            line = f.readline()
            if line == '':
                break

            line = line.split(' ')
            char = line[0]
            embed = list(map(float, line[1:-1]))

            _add_to_vocab(char)
            _add_to_embed(embed)
        print("Embedding loaded successfully.")
        return vocabulary, embeddings


def main():
    # v, e = EmbeddingReader.load("../pre_trained/wiki_char_200.txt")
    # print(v)
    # print(e)
    # print(len(v))

    reader = WeiboReader("../dataset/stc_weibo_train_post_generated_10",
                         "../dataset/stc_weibo_train_response_generated_10",
                         pre_trained_path="../pre_trained/wiki_char_200.txt")
    print(reader.gen_indices_and_lengths("你好啊，玥玥"))
    # reader.set_batch_size(3)
    #
    # for _ in range(4):
    #     print("~~~")
    #     data = reader.next_batch()
    #     print(data.indices)
    #     print(data.lengths)
    #     print(data.weights)
    #     print(reader.config)
    #     # print(reader.vocabulary)


if __name__ == '__main__':
    main()
