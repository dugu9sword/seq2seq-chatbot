class SpToken:
    EOS = "end of a sentence"
    BOS = "begining of a sentence"


class WeiboReader:
    def __init__(self, post_path, response_path):
        f_post = open(post_path)
        f_response = open(response_path)

        self.vocabulary = {}
        add_to_vocab = lambda x: self.vocabulary.setdefault(x, len(self.vocabulary))
        add_to_vocab(SpToken.BOS)
        add_to_vocab(SpToken.EOS)

        posts = f_post.readlines()
        responses = f_response.readlines()
        pairs = zip(posts, responses)
        self.post_indices=[]
        self.response_indices=[]
        self.post_lengths=[]
        self.response_lengths=[]
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
            self.post_indices.append(post_index)
            self.response_indices.append(response_index)



def main():
    reader = WeiboReader("../dataset/stc_weibo_train_post_generated_100",
                         "../dataset/stc_weibo_train_response_generated_100")
    print(reader.vocabulary)
    print(reader.post_indices)


if __name__ == '__main__':
    main()
