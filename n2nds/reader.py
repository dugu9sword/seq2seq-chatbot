from n2nds.config import Config
from n2nds.data import Data
import numpy as np



class SpToken:
    EOS = "|------|end of a sentence"
    BOS = "begining of a sentence"
    NIL = "no word"
    UNK = "unknown word"


class WeiboReader:
    def add_to_vocab(self, word):
        self.vocabulary.setdefault(word, len(self.vocabulary))

    def __init__(self, post_path, response_path):
        self.f_post = open(post_path)
        self.f_response = open(response_path)

        self.vocabulary = {}
        self.add_to_vocab(SpToken.BOS)
        self.add_to_vocab(SpToken.EOS)
        self.add_to_vocab(SpToken.NIL)
        self.add_to_vocab(SpToken.UNK)

        posts = []
        responses = []
        for post in self.f_post.readlines():
            posts.append(post.strip('\n'))
        for response in self.f_response.readlines():
            responses.append(response.strip('\n'))
        self.dataset_pairs = zip(posts, responses)

    def gen_data_and_config_from_dataset(self):
        return self.gen_data_and_config_from_p_r_pairs(self.dataset_pairs)

    def gen_data_and_config_from_p_r_pairs(self, p_r_pairs):
        data = Data()
        data.indices, data.lengths, data.weights = self.gen_length_and_weights(p_r_pairs, True)
        config = Config()
        config.BATCH_SIZE = len(data.indices)
        config.SEQ_SIZE = len(data.indices[0][0])
        config.VOCAB_SIZE = len(self.vocabulary)
        return data, config

    def gen_length_and_weights(self, post_response_pairs, is_first_time=False):
        post_lengths = []
        response_lengths = []
        post_indices = []
        response_indices = []

        MAX_LENGTH = 0

        # Generate indices from words
        for post, response in post_response_pairs:
            post_index = []
            response_index = []

            for utter_index, utter in zip([post_index, response_index], [post, response]):
                for ch in utter:
                    if is_first_time:
                        self.add_to_vocab(ch)
                    if ch in self.vocabulary:
                        utter_index.append(self.vocabulary[ch])
                    else:
                        utter_index.append(self.vocabulary[SpToken.UNK])
                utter_index.append(self.vocabulary[SpToken.EOS])

            post_lengths.append(len(post) + 1)
            response_lengths.append(len(response) + 1)

            MAX_LENGTH = max(len(post)+ 1, len(response)+ 1, MAX_LENGTH)

            post_indices.append(post_index)
            response_indices.append(response_index)

        # Append NIL to the rest of indices
        nil_index = self.vocabulary[SpToken.NIL]
        for post_index in post_indices:
            post_index.extend([nil_index] * (MAX_LENGTH - len(post_index)))
        for response_index in response_indices:
            response_index.extend([nil_index] * (MAX_LENGTH - len(response_index)))

        # Generate lengths
        lengths = list(map(list, zip(post_lengths, response_lengths)))

        # Generate utterances
        utter_indices = list(map(list, zip(post_indices, response_indices)))

        # Generate weights for response
        weights = []
        for response_length in response_lengths:
            weights.append([1] * response_length + [0] * (MAX_LENGTH - response_length))

        return utter_indices, lengths, weights

    def gen_words_from_indices(self, word_indices):
        rev_vocabulary = {v: k for k, v in self.vocabulary.items()}
        return "".join(list(map(rev_vocabulary.get, word_indices[0:-1])))


def main():
    reader = WeiboReader("../dataset/stc_weibo_train_post_generated_100",
                         "../dataset/stc_weibo_train_response_generated_100")
    data, config=reader.gen_data_and_config_from_dataset()
    # print(np.array(data.indices).shape)
    # print(reader.vocabulary)
    # print(reader.post_indices)
    # print(reader.lengths)


if __name__ == '__main__':
    main()
