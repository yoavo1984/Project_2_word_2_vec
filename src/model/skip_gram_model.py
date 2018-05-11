import numpy as np

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

class UniGram(object):
    def __init__(self, corpus):
        """
        A class to represent a uni gram model.
        Args:
            corpus: The corpus we want to generate to uni gram for.
        """
        self.distribution = dict()
        self.corpus = corpus

        self.generate_unigram_distribution()

    def generate_unigram_distribution(self):
        """
        Generates the unigram distribution, counts number of occurrences for each words and
        finishes off with dividing by the total number of words in the corpus.
        """
        num_of_words = self.corpus.get_number_of_words()

        # count the number of times each words appear.
        for word in self.corpus.iterate_words():
            self.distribution[word] = self.distribution.get(word, 0) + 1

        # Divide by the total number of words in the corpus to generate the uni gram distribution
        for key, val in self.distribution.items():
            self.distribution[key] = val / num_of_words

    def ger_random_word(self):
        """
        Returns a random word from the corpus using a unigram distribution.
        Returns:
            (str) : The sampled word.
        """
        words = list(self.distribution.keys())
        p = list(self.distribution.values())

        return np.random.choice(words, p=p)

if __name__ == "__main__":
    class mock_corpus():
        def __init__(self):
            self.str = "This is a long list of words you can use is is is is is is is is is"

        def get_number_of_words(self):
            return len(self.str.split())

        def iterate_words(self):
            for str in self.str.split():
                yield str


    corpus = mock_corpus()
    uni = UniGram(corpus)
    for i in range(10):
        print(uni.ger_random_word())