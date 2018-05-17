import numpy as np

class Corpus():
    """
    This class serves as a cropus of words. 
    The words are stored as id values, were the mapping between words and ids is defined by the
    given dictionary.
    Args:
        sentence_array: The sentences to build our array from
        dictionary: The dicitionary defining the mapping from words to ids.
    """
    def __init__(self, sentence_array, dictionary):

        self.corpus = []

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
        sentence_num = np.random.randint(0, len(self.corpus))
        sentence = self.corpus[sentence_num]

        word_num = np.random.randint(0, len(sentence))
        target = sentence[word_num]

        context = self.get_word_context(sentence_num, word_num, context_size)

        return target, context

    def get_word_context(self, sen_num, word_num, context_size):
        sentence = self.corpus[sen_num]
        context = set()

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

    def __iter__(self):
        for sentence in self.corpus:
            yield sentence

if __name__ == "__main__":
    from src.dataset.wordiddictionary import WordIdDictionary

    sentences = [["this", "is", "a", "sentence"], ["this", "is", "annother", "one"],
                 ["this", "is", "the", "third", "one"]]

    di = WordIdDictionary(sentences)
    corpus = Corpus(sentences, di)

    for sen in corpus:
        print (sen)

    print()

    print(corpus.sample_target_and_context(3))