from n2nds.reader import WeiboReader, Config, SpToken
import tensorflow as tf
from n2nds.seq2seq import Model

class WikiReader():
    def __init__(self, dataset_path, weibo_reader, batch_size):
        self.weibo_reader = weibo_reader

        wiki_data = open(dataset_path)
        self._responses = []
        for line in wiki_data.readlines():
            if len(line) < 50:
                self._responses.append(line.strip('\n'))

        print("Wiki dataset loaded, length %d" % len(self._responses))

        # Generate the config
        self.config = Config()
        self.config.BATCH_SIZE = batch_size
        self.config.SEQ_SIZE = 50
        self.config.VOCAB_SIZE = len(self.weibo_reader.vocabulary)
        self.config.EMBED_SIZE = 200
        self.config.UNIT_SIZE = 200

        # Some variable for batch generation
        self._batch_pointer = 0
        self.dataset_size = len(self._responses)
        self._batch_size = batch_size

    def next_batch(self):
        batch_responses = self._responses[self._batch_pointer:self._batch_pointer + self._batch_size]
        self._batch_pointer = self._batch_pointer + self._batch_size
        if self._batch_pointer + self._batch_size >= self.dataset_size:
            self._batch_pointer = 0

        post_lengths = []
        response_lengths = []
        post_indices = []
        response_indices = []

        # Generate indices from words
        for response in batch_responses:
            post_index = []
            response_index = []

            for ch in response:
                if ch in self.weibo_reader.vocabulary:
                    response_index.append(self.weibo_reader.vocabulary[ch])
                else:
                    response_index.append(self.weibo_reader.vocabulary[SpToken.UNK])
            response_index.append(self.weibo_reader.vocabulary[SpToken.EOS])

            post_lengths.append(0)
            response_lengths.append(len(response) + 1)

            post_indices.append(post_index)
            response_indices.append(response_index)

        # Append NIL to the rest of indices
        nil_index = self.weibo_reader.vocabulary[SpToken.NIL]
        for post_index in post_indices:
            post_index.extend([nil_index] * (self.config.SEQ_SIZE - len(post_index)))
        for response_index in response_indices:
            response_index.extend([nil_index] * (self.config.SEQ_SIZE - len(response_index)))

        # Generate lengths
        lengths = list(map(list, zip(post_lengths, response_lengths)))

        # Generate utterances
        utter_indices = list(map(list, zip(post_indices, response_indices)))

        # Generate weights for response
        weights = []
        for response_length in response_lengths:
            weights.append([1] * response_length + [0] * (self.config.SEQ_SIZE - response_length))

        return utter_indices, lengths, weights


def main():
    weibo_reader = WeiboReader("dataset/stc_weibo_train_post_generated_10",
                               "dataset/stc_weibo_train_response_generated_10",
                               "pre_trained/wiki_char_200.txt",
                               batch_size=-1)
    wiki_reader = WikiReader("dataset/wikipedia_cn_lm",
                             weibo_reader,
                             1)
    print(wiki_reader.next_batch())

    with tf.name_scope("Train"):
        with tf.variable_scope("Model"):
            model = Model(wiki_reader.config, is_train=True,
                          embedding_init_value=weibo_reader.embedding,
                          pre_train_mode=True,
                          num_of_layer=2)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
            valid_model = Model(wiki_reader.config, is_train=False,pre_train_mode=True,
                                num_of_layer=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        utter_indices, lengths, weights=wiki_reader.next_batch()
        print(weibo_reader.gen_words_from_indices(utter_indices[0][1]))
        while True:
            # print(utter_indices)
            # utter_indices, lengths, weights=wiki_reader.next_batch()
            feed_dict = dict()
            loss, op=sess.run([model.cost, model.train_op],feed_dict={
                model.utter_indices:utter_indices,
                model.utter_lengths:lengths,
                model.utter_weights:weights
            })
            print(loss)
            if loss<10:
                break

        feed_dict=dict()
        feed_dict[valid_model.dec_first_input_index]=[0]
        pred = sess.run(valid_model.pred, feed_dict=feed_dict)
        print(weibo_reader.gen_words_from_indices(pred))

if __name__ == '__main__':
    main()
