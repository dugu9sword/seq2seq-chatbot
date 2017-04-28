class Data:
    def __init__(self):
        # Should be [batch_size * 2 * sequence_size]
        self.indices = [[[2, 3, 4, 5, 0],
                         [4, 1, 1, 2, 0]],
                        [[1, 2, 3, 0, 0],
                         [3, 4, 5, 3, 0]]]
        # Should be [batch_size * 2]
        self.lengths = [[5, 5],
                        [5, 5]]
        # Should be [batch_size * sequence_size]
        self.weights = [[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]

