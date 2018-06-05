import sys
import time
import pickle

from src.dataset.dataset import Dataset
from src.dataset.sentence_assignments import SentenceAssigner
from src.evaluations.word2vec_evaluations import *
from src.learning.gradient_descent import GradientDescent
from src.learning.learning_hyperparameters import LearningHyperparameters
from src.model.model_hyperparameters import Hyperparameters
from src.model.skip_gram_model import SkipGramModel

POSSIBLE_D = [10, 50, 100, 150, 200, 250, 300]
POSSIBLE_CONTEXTS = [1,2,3,4,5]

def hyper_parameters_by_configuration(file_name="configuration"):
    with open(file_name) as config:
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
    sk_model = SkipGramModel(m_hyperparameter, train_corpus)

    # Setting up learner.
    print("- Setting up learner.")
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

def test_evaluations():
    train_corpus, test_corpus = get_train_and_test()
    m_hyperparameters, l_hyperparameters = hyper_parameters_by_configuration()

    sk_model = SkipGramModel(m_hyperparameters, train_corpus)
    sk_model.load_model()

    print("-- Analogy Solver --")
    analogy_solver(sk_model , "man", "woman", "king")
    analogy_solver(sk_model , "good", "great", "bad")
    # analogy_solver(sk_model , "good", "great", "bad")

    # scatter_input_in_2D(sk_model, ["man", "woman", "king", "queen"])

    # print("-- Dog --")
    # predict_most_likely_context(sk_model, "dog")
    # print("-- Cat --")
    # predict_most_likely_context(sk_model, "cat")
    # print("-- Princess --")
    # predict_most_likely_context(sk_model, "princess")

    predict_most_likely_input(sk_model, "Where did the go to today".split())


def deliverable_two():
    print("- Setting up dataset")
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)
    train_corpus = dataset.train_corpus
    test_corpus = dataset.test_corpus

    # Get hyperparameters from configuration file
    m_hyperparameter, l_hyperparameters = hyper_parameters_by_configuration("configuration_2")

    # Setting up model.
    print("- Setting up model")
    data_object = {}

    # Prepare pickle object
    data = {
        "time": {
            "hyper": "This are the set of parameters",
            "running_time": [],
            "d": []
        },
        "likelihood": {
            "hyper": "This are the set of parameters",
            "train": [],
            "test": [],
            "d": []
        }
    }
    data_object["time"] = {}
    data_object["likelihood"] = {}
    training_time = []
    for d in POSSIBLE_D:
        print("Learning with d = {}".format(d))
        m_hyperparameter.d = d
        sk_model = SkipGramModel(m_hyperparameter, train_corpus)

        # Setting up learner.
        print("- Setting up learner.")
        learner = GradientDescent(l_hyperparameters)

        start = time.time()
        learner.learnParamsUsingSGD(sk_model, train_corpus, test_corpus)

        train_liklihood = learner.compute_model_likelihood(sk_model, train_corpus)
        test_liklihood = learner.compute_model_likelihood(sk_model, test_corpus)

        data["time"]["running_time"].append(time.time() - start)
        data["time"]["d"].append(d)
        data["likelihood"]["train"].append(train_liklihood)
        data["likelihood"]["test"].append(test_liklihood)
        data["likelihood"]["d"].append(d)

    print("- Finished learning")
    sk_model.save_model()
    pickle.dump(data, open("d2", "wb"))

def deliverable_three():
    print("- Setting up dataset")
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)
    train_corpus = dataset.train_corpus
    test_corpus = dataset.test_corpus

    # Get hyperparameters from configuration file
    m_hyperparameter, l_hyperparameters = hyper_parameters_by_configuration("configuration_2")

    # Setting up model.
    print("- Setting up model")
    data_object = {}

    # Prepare pickle object
    data = {
        "time": {
            "hyper": "This are the set of parameters",
            "running_time": [],
            "context_size": []
        },
        "likelihood": {
            "hyper": "This are the set of parameters",
            "train": [],
            "test": [],
            "context_size": []
        }
    }
    data_object["time"] = {}
    data_object["likelihood"] = {}
    training_time = []
    for context_size in POSSIBLE_CONTEXTS:
        print("Learning with context size = {}".format(context_size))
        m_hyperparameter.context_size = context_size
        sk_model = SkipGramModel(m_hyperparameter, train_corpus)

        # Setting up learner.
        print("- Setting up learner.")
        learner = GradientDescent(l_hyperparameters)

        start = time.time()
        learner.learnParamsUsingSGD(sk_model, train_corpus, test_corpus)

        train_liklihood = learner.compute_model_likelihood(sk_model, train_corpus)
        test_liklihood = learner.compute_model_likelihood(sk_model, test_corpus)

        data["time"]["running_time"].append(time.time() - start)
        data["time"]["context_size"].append(context_size)
        data["likelihood"]["train"].append(train_liklihood)
        data["likelihood"]["test"].append(test_liklihood)
        data["likelihood"]["context_size"].append(context_size)

    print("- Finished learning")
    sk_model.save_model()
    pickle.dump(data, open("d3", "wb"))

def main():
    print("- Setting up dataset")
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)
    train_corpus = dataset.train_corpus
    test_corpus = dataset.test_corpus
    # Get hyperparameters from configuration file
    m_hyperparameter, l_hyperparameters = hyper_parameters_by_configuration()

    # Setting up model.
    print("- Setting up model")
    sk_model = SkipGramModel(m_hyperparameter, train_corpus)

    # Setting up learner.
    print("- Setting up learner.")
    learner = GradientDescent(l_hyperparameters)
    learner.learnParamsUsingSGD(sk_model, train_corpus, test_corpus)

    print("- Finished learning")
    sk_model.save_model()

if __name__ == "__main__":
    deliverable_three()


