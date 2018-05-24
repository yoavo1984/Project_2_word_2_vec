class Hyperparameters(object):

    def __init__(self, win_size, d, k, decay_rate, noise_d, seed):
        """
        This class represent a model hyperparameters. 
        Args:
            win_size(int): The size of the context window. 
            d(int): The size of the vector representation of each word.
            k(int): Number of "noncontext" negative words. 
            decay_rate(int): Number of iterations between reductions of the learning rate.
            noise_d(object): Choice of noise distribution.
            seed(int): Random seed for governing randomness.
        """
        self.win_size = win_size
        self.d = d
        self.k = k
        self.decay_rate = decay_rate
        self.noise_d = noise_d
        self.seed = seed
