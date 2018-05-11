class SkipGramModel(object):
    def __init__(self, hyperparameters, training_data):
        pass
        # Create a vector for each word in the training data (u_i,, v_i)

        # Sample each vector from a multivariative Gaussian

        # Normalize each vector (L2 Norm).

    def sample_word(self, alpha=1):
        """
        Sample words from the vocabulary using a unigram to the alpha distribution.
        Args:
            alpha(float): The power to apply on the unigram distribution  

        Returns:
            A word sampled using the unigram to power alpha distribution
        """
        pass

    def sample_k_words(self):
        """
        Sample k words according to the hyperparameters noise distribution.
         k is also defined in the hyperparameters object.
        Returns:
            (list) k words sampled from the hyperparameters noise distribution
        """
        pass

        def compute_log_probability(word_i, word_j, words_k):
            """
            Computes the log probability of word_i given word_j and K negative sampled
            words.
            Args:
                word_i: Target word. 
                word_j: Given context word
                words_k: K sampled negative words

            Returns:
                The log probablity of the given Target word given the context word and K
                negative sampled words
            """
            return 0.0