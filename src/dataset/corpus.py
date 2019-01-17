import numpy as np


class Corpus():
    """
    This class serves as a corpus of words. 
    The words are stored as id values, were the mapping between words and ids is defined by the
    given dictionary.
    Args:
        sentence_array: The sentences to build our array from
        dictionary: The dictionary defining the mapping from words to ids.
    """
    def __init__(self, sentence_array, dictionary):

        self.corpus = []
        self.word_dictionary = dictionary

        # Translate words into id's.
        for index, sentence in enumerate(sentence_array):
            self.corpus.append([])
            for word in sentence:
                word_id = dictionary.get_id_by_word(word)
                self.corpus[index].append(word_id)

    def sample_target_and_context(self, context_size):
        """
        Samples a a target word and a context word uniformly from the corpus, according to the context size.
        Args:
            context_size: The size of the context window.

        Returns: a word (target) and a set of words(contexts).

        """

        # Sample a random sentence.
        sentence_num = np.random.randint(0, len(self.corpus))
        sentence = self.corpus[sentence_num]

        # Sample a random word.
        word_num = np.random.randint(0, len(sentence))
        target = sentence[word_num]

        # Get the words context window.
        context = self.get_word_context(sentence_num, word_num, context_size)

        return target, context

    def get_word_context(self, sen_num, word_num, context_size):
        """
        Pulls the given word context window.
        Args:
            sen_num: The sentence number from which to pull the context
            word_num: The input word. The context will be around it.
            context_size: The size of the context window.

        Returns:
            The context words around the given input.
        """
        sentence = self.corpus[sen_num]
        context = set()

        # Take all words around the given word.
        for i in range(1, context_size + 1):
            right_side = word_num + i
            left_side = word_num - i
            if right_side < len(sentence):
                context.add(sentence[right_side])
            if left_side >= 0:
                context.add(sentence[left_side])

        return context

    def iterate_words(self):
        """
        Iterate over all the words in the corpus one by one.
        Returns: Yields the words in the dictionary one by one.

        """
        for sentence in self.corpus:
            for word in sentence:
                yield word

    def iterate_target_context(self, context_size):
        """
        Iterate over all the target - context pair words in the corpus.
        Args:
            context_size: Size of context window.

        Returns:
            Yield context-target words 1 by 1.
        """
        for sentence_index, sentence in enumerate(self.corpus):
            for word_index, word in enumerate(sentence):
                context_words = self.get_word_context(sentence_index, word_index, context_size)
                yield word, context_words

    def get_words_count(self):
        """
        Return the amount of words in the corpus.
        Returns:
            The amount of words in the corpus (including repetition)
        """
        count = 0
        for _ in self.iterate_words():
            count += 1

        return count

    def get_unique_words_count(self):
        """
        Get the amount of unique words in the corpus
        Returns:
            The amount of words in the corpus (excluding repetition)
        """
        return self.word_dictionary.get_dictionary_length()

    def get_word_by_index(self, index):
        return self.word_dictionary[index]

    def __iter__(self):
        for sentence in self.corpus:
            yield sentence

    def __getitem__(self, value):
        return self.corpus[value]
