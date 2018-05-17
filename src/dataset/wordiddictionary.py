class WordIdDictionary():
    """
    This serves as a word "dictionary" where by dictionary we mean a mapping from 
      a word to a unique id.
    Args:
        sentence_array: 
    """
    def __init__(self, sentence_array):
        # A two way dictionary (implementation below)
        self.twoWayDict = TwoWayDict()

        # The words id counter. To give unique id's to words.
        word_id = 0

        # Iterate over words and set their id value.
        for sentence in sentence_array:
            for word in sentence:
                if word not in self.twoWayDict:
                    self.twoWayDict[word] = word_id
                    word_id += 1

    def get_word_by_id(self, id):
        if id in self.twoWayDict:
            return self.twoWayDict[id]
        else:
            return -1

    def get_id_by_word(self, word):
        if word in self.twoWayDict:
            return self.twoWayDict[word]
        else:
            return -1


class TwoWayDict(dict):
    """
    An implementation of a 2 way dictionary gives us an easy abstraction
    for storing elements in a dictionary and retrieving them using both keys and values. 
    """
    def __setitem__(self, key, value):

        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


if __name__ == "__main__":
    sentences = [["this", "is", "a", "sentence"], ["this", "is", "annother", "one"], ["this", "is", "the", "third", "one"]]
    di = WordIdDictionary(sentences)

    print(di.get_id_by_word('is'))
    print(di.get_id_by_word('a'))
    print(di.get_id_by_word('one'))
    print(di.get_id_by_word('bbb'))
    print("---")
    print(di.get_word_by_id(1))
    print(di.get_word_by_id(2))
    print(di.get_word_by_id(5))