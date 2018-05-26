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
        self.context_size = int(context_size)
        self.d = int(d)
        self.k = int(k)
        self.noise_d = float(noise_d)
        self.seed = int(seed)
