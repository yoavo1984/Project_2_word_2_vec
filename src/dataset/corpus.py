class Corpus():
    def __init__(self, sentence_array, dictionary):
        self.corpus = []

        for index, sentence in enumerate(sentence_array):
            self.corpus.append([])
            for word in sentence:
                word_id = dictionary.get_id_by_word(word)
                self.corpus[index].append(word_id)

    def iterate_words(self):
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

    for sentence in corpus:
        print (sentence)