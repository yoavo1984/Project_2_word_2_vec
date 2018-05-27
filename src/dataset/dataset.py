# Standard libraries
import re

# Local imports
from src.dataset.wordiddictionary import WordIdDictionary
from src.dataset.corpus import Corpus

# Test Imports
from src.dataset.sentence_assignments import SentenceAssigner

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
                if len(words) > 0:
                    sentences.append(words)

        return sentences

    def set_test_train_sentences(self):
        for index, sentence in enumerate(self.sentences):
            assignment = self.assigner.assign(index)
            if assignment == 1:
                self.train_sentences.append(sentence)
            if assignment == 2:
                self.test_sentences.append(sentence)

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

    def get_dictionary(self):
        return self.dictionary

    @staticmethod
    def preprocess(line):
        # Lowercase
        line = line.lower()

        # Removing non ascii characters
        line = line.encode("ascii", errors="ignore").decode()

        # Removing non-alphanumeric characters
        pattern = re.compile(r'([^\s\w]|_)+')
        line = re.sub(pattern, '', line)

        # Generating list of words
        words = [x for x in line.split()]

        # Removing first word as it is just the index of the sentence
        words = words[1:]

        # Remove words smaller then 2
        words = [x for x in words if len(x) > 2]

        return words

def get_dataset():
    sc = SentenceAssigner("../../data/datasetSplit.txt")
    dataset = Dataset("../../data/datasetSentences.txt", sc)

    return dataset
if __name__ == "__main__":
    sc = SentenceAssigner("../../data/datasetSplit.txt")
    dataset = Dataset("../../data/datasetSentences.txt", sc)

    train_corpus = dataset.train_corpus