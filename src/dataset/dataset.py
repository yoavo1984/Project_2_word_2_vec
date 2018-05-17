# Standard libraries
import re

# Local imports
from src.dataset.wordiddictionary import WordIdDictionary
from src.dataset.corpus import Corpus

class Dataset(object):
    """
    This class is in charge of holding the entire dataset: Train and Test instances.
         Args:
            data_path(str): The path for datasetSentence.txt file.
            sentence_aligner(object): An object in charge of assigning sentences to train and test
            by there id.
    """
    def __init__(self, data_path, sentence_assigner):
        # Create the corpus for the given data.
        self.corpus = set()
        self.generate_corpus(data_path)

        self.assigner = sentence_assigner

        self.sentences = self.generate_sentences(data_path)

        # Building sentences arrays from the files.
        self.train_sentences = []
        self.test_sentences = []
        self.set_test_train_sentences()

        # Building the train and test corpora.
        self.dictionary = WordIdDictionary(self.train_sentences)
        self.train_corpus = Corpus(self.train_sentences, self.dictionary)
        self.test_corpus = Corpus(self.test_sentences, self.dictionary)

    def generate_sentences(self, data_path):
        """
        Creates the sentence array from the given data file.
        Args:
            data_path: The path of the file from which to build the sentences.

        """
        sentences = []
        with open(data_path) as file:
            file.readline()

            for line in file:
                words = self.preprocess(line)
                sentences.append(words)

        return sentences

    def set_test_train_sentences(self):
        for index, sentence in enumerate(self.sentences):
            assignment = self.assigner(index)
            if assignment == 1:
                self.train_sentences.add(sentence)
            if assignment == 2:
                self.test_sentences.add(sentence)

    def generate_corpus(self, data_path):
        """
        Generate the corpus of the dataset. 
        Implementation involves inserting all the words into a set.
        """
        with open(data_path) as file:
            file.readline()

            for line in file:
                # Getting the words in the line
                words = self.preprocess(line)

                # Add all words to our set.
                self.corpus.update(words)

    @staticmethod
    def preprocess(line):
        # Lowercase
        line = line.lower()

        # Removing non ascii characters
        line = line.encode("ascii", errors="ignore").decode()

        # Removing non-alphanumeric characters
        pattern = re.compile(r'([^\s\w]|_)+')
        line = re.sub(pattern, '', line)

        # Generating list of words and removing words shorter than 3
        words = [x for x in line.split() if len(x) > 2]

        return words