from src.dataset.dataset import Dataset
from src.dataset.sentence_assignments import SentenceAssigner
from src.learning.gradient_descent import GradientDescent
from src.learning.learning_hyperparameters import LearningHyperparameters
from src.model.model_hyperparameters import Hyperparameters
from src.model.skip_gram_model import SkipGramModel


def run_sgd():
    print("- Setting up dataset")
    sc = SentenceAssigner("../data/datasetSplit.txt")
    dataset = Dataset("../data/datasetSentences.txt", sc)
    train_corpus = dataset.train_corpus

    # Setting up model.
    print("- Setting up model")
    hyperparameters = Hyperparameters(2, 10, 5, 1, 123)
    sk_model = SkipGramModel(hyperparameters, train_corpus)

    # Setting up learner.
    print("- Setting up learner.")
    l_hyperparameters = LearningHyperparameters(0.25, 50, 100)
    learner = GradientDescent(l_hyperparameters)
    learner.learnParamsUsingSGD(sk_model, train_corpus)

if __name__ == "__main__":
    run_sgd()