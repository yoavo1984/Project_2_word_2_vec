import sys
import time

from src.dataset.dataset import Dataset
from src.dataset.sentence_assignments import SentenceAssigner
from src.learning.gradient_descent import GradientDescent
from src.learning.learning_hyperparameters import LearningHyperparameters
from src.model.model_hyperparameters import Hyperparameters
from src.model.skip_gram_model import SkipGramModel

def hyper_parameters_by_configuration():
    with open('configuration') as config:
        lines = config.read().splitlines()
        # lines = config.readlines()
        # lines = [line[:-1] for line in lines]

        # Remove comments and empty lines.
        lines = [x for x in lines if len(x) > 0 and x[0] != "#"]

        for line in lines:
            parameters = line.split(":")
            hyperparameters = parameters[1:]

            if parameters[0] == "model":
                model_hyperparameters = Hyperparameters(*hyperparameters)

            if parameters[0] == "learning":
                learning_hyperparameters = LearningHyperparameters(*hyperparameters)

        if model_hyperparameters is None or learning_hyperparameters is None:
            sys.exit("Problem with configuration file.")

        return model_hyperparameters, learning_hyperparameters

def run_sgd():
    print("- Setting up dataset")
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)
    train_corpus = dataset.train_corpus
    test_corpus = dataset.test_corpus

    # Get hyperparameters from configuration file
    m_hyperparameter, l_hyperparameters = hyper_parameters_by_configuration()

    # Setting up model.
    print("- Setting up model")
    hyperparameters = Hyperparameters(2, 10, 5, 1, 123)
    sk_model = SkipGramModel(hyperparameters, train_corpus)

    # Setting up learner.
    print("- Setting up learner.")
    l_hyperparameters = LearningHyperparameters(0.25, 50, 100)
    learner = GradientDescent(l_hyperparameters)
    learner.learnParamsUsingSGD(sk_model, train_corpus, test_corpus)

    print("- Finished learning")
    sk_model.save_model()


def get_train_and_test():
    # Read the data and split into train and test.
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)

    train_corpus = dataset.train_corpus
    test_corpus = dataset.test_corpus

    return train_corpus, test_corpus


def main():
    # Read the data and split into train and test.
    train_corpus, test_corpus = get_train_and_test()

    # Set hyperparameters based on configuration file.
    m_hyperparameters, l_hyperparameters = hyper_parameters_by_configuration()

    # Learn model and start the timer
    start = time.time()

    sk_model = SkipGramModel(m_hyperparameters, train_corpus)
    learner = GradientDescent(l_hyperparameters)
    learner.learnParamsUsingSGD(sk_model, train_corpus, test_corpus)

    # Output hyperparameters and log-likelihoods to file.
    with open("output", "w") as output:
        # TODO add relevant attributes.
        output.write("hyperparameters")
        output.write("log likelihood")
        output.write("Learning Time : {}".format(time.time() - start))

if __name__ == "__main__":
    run_sgd()