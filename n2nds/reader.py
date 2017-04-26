import copy

class SpToken:
    EOS = "end of a sentence"
    BOS = "begining of a sentence"
    NIL = "no word"


class WeiboReader:
    def __init__(self, post_path, response_path):
        f_post = open(post_path)
        f_response = open(response_path)

        self.vocabulary = {}
        add_to_vocab = lambda x: self.vocabulary.setdefault(x, len(self.vocabulary))
        add_to_vocab(SpToken.BOS)
        add_to_vocab(SpToken.EOS)
        add_to_vocab(SpToken.NIL)

        posts = f_post.readlines()
        responses = f_response.readlines()
        pairs = zip(posts, responses)
        self.post_indices = []
        self.response_indices = []
        self.post_lengths = []
        self.response_lengths = []
        self.weights = []
        MAX = 0
        for post, response in pairs:
            post_index = []
            response_index = []
            for ch in post:
                if ch != '\n':
                    add_to_vocab(ch)
                    post_index.append(self.vocabulary[ch])
                else:
                    post_index.append(self.vocabulary[SpToken.EOS])
            for ch in response:
                if ch != '\n':
                    add_to_vocab(ch)
                    response_index.append(self.vocabulary[ch])
                else:
                    response_index.append(self.vocabulary[SpToken.EOS])
            self.post_lengths.append(len(post))
            self.response_lengths.append(len(response))
            MAX = MAX if MAX>len(post) else len(post)
            MAX = MAX if MAX>len(response) else len(response)
            self.post_indices.append(post_index)
            self.response_indices.append(response_index)
        nil_index=self.vocabulary[SpToken.NIL]
        for post_index in self.post_indices:
            post_index.extend([nil_index]*(MAX-len(post_index)))
        for response_index in self.response_indices:
            response_index.extend([nil_index]*(MAX-len(response_index)))
        # ziplist=lambda a, b: map(list, zip(a, b))
        self.lengths = list(map(list,zip(self.post_lengths, self.response_lengths)))
        self.utterances = list(map(list,zip(self.post_indices, self.response_indices)))
        for response_length in self.response_lengths:
            self.weights.append([1]*response_length+[0]*(MAX-response_length))
        print(MAX)
        print(self.utterances)
        print(self.lengths)
        print(self.weights)

    def gen_data(self):
        pass

def main():
    reader = WeiboReader("../dataset/stc_weibo_train_post_generated_100",
                         "../dataset/stc_weibo_train_response_generated_100")
    # print(reader.vocabulary)
    # print(reader.post_indices)
    # print(reader.lengths)


if __name__ == '__main__':
    main()
