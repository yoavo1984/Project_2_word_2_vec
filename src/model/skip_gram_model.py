import numpy as np

class SkipGramModel(object):
    def __init__(self, hyperparameters, training_corpus):
        self.corpus = training_corpus

        # Building the noise distribution.
        self.unigram = UniGram(training_corpus)
        self.unigram.set_alpha(hyperparameters.noise_d)

        self.k = hyperparameters.k

        # Create a vector for each word in the training data (u_i, v_i), and Sample each vector from a
        # multivariate Gaussian
        number_of_words = training_corpus.get_words_count()
        self.target_vectors = np.random.normal(0, 0.01, size=(number_of_words, hyperparameters.d))
        self.context_vectors = np.random.normal(0, 0.01, size=(number_of_words, hyperparameters.d))

        # Normalize each vector (L2 Norm).
        self.normalize_vectors()

    def normalize_vectors(self):
        """
        Normalize all the model vectors(target and context) to have a norm of 1.
        Returns:

        """
        target_norms = np.linalg.norm(self.target_vectors, axis=1, keepdims=True)
        self.target_vectors = self.target_vectors / target_norms

        context_norms = np.linalg.norm(self.context_vectors, axis=1, keepdims=True)
        self.context_vectors = self.context_vectors / context_norms


    def sample_word(self, alpha=1):
        """
        Sample words from the vocabulary using a unigram to the alpha distribution.
        Args:
            alpha(float): The power to apply on the unigram distribution  

        Returns:
            A word sampled using the unigram to power alpha distribution
        """
        return self.unigram.sample_word()

    def sample_k_words(self):
        """
        Sample k words according to the hyperparameters noise distribution.
         k is also defined in the hyperparameters object.
        Returns:
            (list) k words sampled from the hyperparameters noise distribution
        """
        samples = []
        for i in range (self.k):
            samples.append(self.sample_word())

        return samples

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

    def reset_probabilites(self):
        self.generate_unigram_distribution()

    def sample_word(self):
        """
        Returns a random word from the corpus using a unigram distribution.
        Returns:
            (str) : The sampled word.
        """
        words = list(self.distribution.keys())
        p = list(self.distribution.values())

        return np.random.choice(words, p=p)

    def set_alpha(self, alpha):
        self.reset_probabilites()
        probabilites_sum = 0

        for key,val in self.distribution.items():
            new_value = pow(val, alpha)
            self.distribution[key] = new_value
            probabilites_sum += new_value

        for key,val in self.distribution.items():
            self.distribution[key] = val / probabilites_sum


if __name__ == "__main__":
    class mock_corpus():
        def __init__(self):
            self.str = "is is is is is is is is is is is is one"

        def get_number_of_words(self):
            return len(self.str.split())

        def iterate_words(self):
            for str in self.str.split():
                yield str


    corpus = mock_corpus()
    uni = UniGram(corpus)

    print(" -- Unigram Sampling --")
    for i in range(10):
        print(uni.sample_word())

    uni.set_alpha(0)

    print("\n -- Uniform Sampling --")
    for i in range(10):
        print(uni.sample_word())

    uni.set_alpha(0.5)
    print("\n -- alpha=0.5 Sampling --")
    for i in range(10):
        print(uni.sample_word())