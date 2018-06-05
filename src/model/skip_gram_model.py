# Standard Libraries
import numpy as np
import pickle

# Test imports
from src.model.model_hyperparameters import Hyperparameters
from src.dataset import dataset


class SkipGramModel(object):
    def __init__(self, hyperparameters, training_corpus):
        self.corpus = training_corpus
        self.word_dictionary = training_corpus.word_dictionary

        # Building the noise distribution.
        self.unigram = UniGram(training_corpus)
        self.unigram.set_alpha(hyperparameters.noise_d)

        self.k = hyperparameters.k

        # Create a vector for each word in the training data (u_i, v_i), and Sample each vector from a
        # multivariate Gaussian
        number_of_words = training_corpus.get_unique_words_count()
        self.target_vectors = np.random.normal(0, 1, size=(number_of_words, hyperparameters.d))
        self.context_vectors = np.random.normal(0, 1, size=(number_of_words, hyperparameters.d))

        # Normalize each vector (L2 Norm).
        self.normalize_vectors()

        # Save the context size
        self.context_size = hyperparameters.context_size

    def normalize_vectors(self):
        """
        Normalize all the model vectors(target and context) to have a norm of 1.
        Returns:

        """
        target_norms = np.linalg.norm(self.target_vectors, axis=1, keepdims=True)
        self.target_vectors = self.target_vectors / target_norms

        context_norms = np.linalg.norm(self.context_vectors, axis=1, keepdims=True)
        self.context_vectors = self.context_vectors / context_norms


    def sample_word(self, alpha=1, k=0):
        """
        Sample words from the vocabulary using a unigram to the alpha distribution.
        Args:
            alpha(float): The power to apply on the unigram distribution  

        Returns:
            A word sampled using the unigram to power alpha distribution
        """
        return self.unigram.sample_word(k=k)

    def sample_k_words(self):
        """
        Sample k words according to the hyperparameters noise distribution.
         k is also defined in the hyperparameters object.
        Returns:
            (list) k words sampled from the hyperparameters noise distribution
        """
        samples = self.sample_word(k=self.k)

        return samples

    def compute_log_probability(self, word_i, word_j, words_k):
        """
        Computes the log probability of word_i given word_j and K negative sampled
        words.
        Args:
            word_i: Context word. 
            word_j: Given target word
            words_k: K sampled negative context words

        Returns:
            The log probablity of the given Target word given the context word and K
            negative sampled words
        """
        context_target_sigmoid = self.compute_sigmoid(word_i, word_j)
        sum = np.log(context_target_sigmoid)
        k = len(words_k)

        test = 0
        for word in words_k:
            sum += (1/k) * np.log(1 - self.compute_sigmoid(word, word_j))

        return sum

    def compute_softmax(self, context, target):
        # Get relevant vectors.
        context_vector = self.context_vectors[context]
        target_vector = self.target_vectors[target]

        # Compute the dot product between the context vector and the target vector
        target_context_dot_prod = np.exp(np.dot(target_vector, context_vector))

        # Compute the sum of dot products of the target vector with all the contexts.
        target_all_sum = sum(np.exp(self.context_vectors.dot(target_vector)))

        # Compute the log to get the probability.
        softmax = (target_context_dot_prod / target_all_sum)

        return softmax

    def compute_sigmoid(self, context, target):
        return 1  / (1 +  np.exp (-1 * np.dot(self.context_vectors[context],
                                             self.target_vectors[target])) )

    def update_parameters(self, target_gradients, context_gradients):
        # Add the gradients.
        self.target_vectors += target_gradients
        self.context_vectors += context_gradients

        # Normalize the vectors.
        self.normalize_vectors()

    def get_context_vector(self, context):
        if isinstance(context, str):
            context = self.get_index_by_word(context)
        return self.context_vectors[context]

    def get_target_vector(self, target):
        if isinstance(target, str):
            target = self.get_index_by_word(target)

        return self.target_vectors[target]

    def get_latent_space_size(self):
        return len(self.target_vectors[0])

    def get_matrix_size(self):
        # TODO rename?
        return self.target_vectors.shape

    def save_model(self):
        np.savetxt("model_target", self.target_vectors, delimiter=",")
        np.savetxt("model_context", self.context_vectors, delimiter=",")

    def load_model(self):
        self.context_vectors = np.loadtxt("model_context", delimiter=",")
        self.target_vectors  = np.loadtxt("model_target", delimiter=",")

    def find_most_similar_vector(self, to_vector, context=False):
        if context:
            multiplication = np.matmul(self.context_vectors, to_vector.T)
        else:
            multiplication = np.matmul(self.target_vectors, to_vector.T)

        max_words = np.argpartition(multiplication, -10)[-10:]

        # Return the relevant word
        return max_words

    def get_word_by_index(self, index):
        return self.word_dictionary.get_word_by_id(index)

    def get_index_by_word(self, word):
        return self.word_dictionary.get_id_by_word(word)




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

        self.sample_bank = []

    def generate_unigram_distribution(self):
        """
        Generates the unigram distribution, counts number of occurrences for each words and
        finishes off with dividing by the total number of words in the corpus.
        """
        num_of_words = self.corpus.get_words_count()

        # count the number of times each words appear.
        for word in self.corpus.iterate_words():
            self.distribution[word] = self.distribution.get(word, 0) + 1

        # Divide by the total number of words in the corpus to generate the uni gram distribution
        for key, val in self.distribution.items():
            self.distribution[key] = val / num_of_words

    def reset_probabilites(self):
        self.generate_unigram_distribution()

    def sample_word(self, k):
        """
        Returns a random word from the corpus using a unigram distribution.
        Returns:
            (str) : The sampled word.
        """
        if k==0:
            return []

        if len(self.sample_bank) < k:
            self.refill_sample_bank()

        samples = self.sample_bank[:k]
        self.sample_bank = self.sample_bank[k:]

        return samples

    def refill_sample_bank(self):
        words = list(self.distribution.keys())
        p = list(self.distribution.values())

        new_samples = np.random.choice(words, p=p, size=100000, replace=True)

        self.sample_bank.extend(new_samples)

    def set_alpha(self, alpha):
        self.reset_probabilites()
        probabilities_sum = 0

        for key,val in self.distribution.items():
            new_value = pow(val, alpha)
            self.distribution[key] = new_value
            probabilities_sum += new_value

        for key,val in self.distribution.items():
            self.distribution[key] = val / probabilities_sum


if __name__ == "__main__":
    hyperparameters = Hyperparameters(2, 10, 5, 10, 1, 123)
    np.random.seed(123)
    dataset_class = dataset.get_dataset()
    train_corpus = dataset_class.train_corpus

    sk_model = SkipGramModel(hyperparameters, train_corpus)

    for i in range(10):
        k_words = sk_model.sample_k_words()
        target, contexts = train_corpus.sample_target_and_context(3)

        print(sk_model.compute_log_probability(contexts.pop(), target, k_words))