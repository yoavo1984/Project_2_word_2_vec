class Hyperparameters(object):

    def __init__(self, context_size, d, k, noise_d, seed):
        """
        This class represent a model hyperparameters. 
        Args:
            context_size(int): The size of the context window. 
            d(int): The size of the vector representation of each word.
            k(int): Number of "non-context" negative words. 
            noise_d(object): Choice of noise distribution.
            seed(int): Random seed for governing randomness.
        """
        self.context_size = context_size
        self.d = d
        self.k = k
        self.noise_d = noise_d
        self.seed = seed
