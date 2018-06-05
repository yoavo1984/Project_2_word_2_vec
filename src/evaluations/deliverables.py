# Standard libraries.
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Local libraries
from src.evaluations.word2vec_evaluations import *


def deliverable_one(model):
    """
    Plot the log-likelihood train and test as a function of training iteration
    Args:
        model: 

    Returns:

    """
    d1_data = pickle.load(open("d1_test", "rb"))

    iterations = d1_data["iterations"]
    train_error = d1_data['train']
    test_error = d1_data['test']
    hyper_string = d1_data["hyper"]


    plt.xlabel("Iteration")
    plt.ylabel("Mean Log-Likelihood")

    plt.title(hyper_string)
    plt.plot(iterations, train_error)
    plt.plot(iterations, test_error)

    plt.legend(["Train Log-Likelihood" ,"Test Log-Likelihood"])
    plt.show()

    plt.clf()


def deliverable_two(model):
    """
    Plot both training time and train and test (mean) log likelihood as a function of d.
    * Two separate plots.
    Args:
        model: 

    Returns:

    """

    d2_data = pickle.load(open("d2", "rb"))

    # Plot training time vs d
    time_data = d2_data["time"]
    hyperparameters = "hyper"
    running_time = time_data["running_time"]
    d = time_data["d"]

    plt.xlabel("d")
    plt.ylabel("Running Time")

    plt.title(hyperparameters)
    plt.plot(d, running_time)

    plt.show()
    plt.clf()

    # Plot mean log-likelihood vs d
    likelihood_data = d2_data["likelihood"]
    hyperparameters = "hyper"
    train = likelihood_data["train"]
    test = likelihood_data["test"]
    d = likelihood_data["d"]

    plt.xlabel("d")
    plt.ylabel("Mean Log-Likelihood")

    plt.title(hyperparameters)
    plt.plot(d, train)
    plt.plot(d, test)
    plt.legend(["Train Log-Likelihood", "Test Log-Likelihood"])

    plt.show()
    plt.clf()


def deliverable_three(model):
    """
    Plot both training time and train and test (mean) log likelihood as a function of d.
    * Two separate plots.
    Args:
        model: 

    Returns:

    """
    d3_data = pickle.load(open("d3_test", "rb"))

    # Plot training time vs d
    time_data = d3_data["time"]
    hyperparameters = time_data["hyper"]
    running_time = time_data["running_time"]
    batch_size = time_data["batch_size"]

    plt.xlabel("Batch Size")
    plt.ylabel("Running Time")

    plt.title(hyperparameters)
    plt.plot(batch_size, running_time)

    plt.show()
    plt.clf()

    # Plot mean log-likelihood vs d
    likelihood_data = d3_data["likelihood"]
    hyperparameters = likelihood_data["hyper"]
    train = likelihood_data["train"]
    test = likelihood_data["test"]
    batch_size = likelihood_data["batch_size"]

    plt.xlabel("Batch Size")
    plt.ylabel("Mean Log-Likelihood")

    plt.title(hyperparameters)
    plt.plot(batch_size, train)
    plt.plot(batch_size, test)
    plt.legend(["Train Log-Likelihood", "Test Log-Likelihood"])

    plt.show()
    plt.clf()

def deliverable_four(model):
    given_words = ["good", "bad", "lame", "cool", "exciting"]
    for word in given_words:
        # Find 10 most likely contexts
        matching_contexts = predict_most_likely_context(model, word, 10)

        #
        pass

def deliverable_five(model):
    sentence_one = ["the", "movie", "was", "surprisingly", ""]
    sentence_two = ["", "was", "really", "disappointing"]
    sentence_three = ["Knowing", "that", "she", "", "was", "the", "best"]
    print("five")
    pass

def deliverable_six(model):
    print("six")
    pass


DELIVERABLES = [deliverable_one,
                deliverable_two,
                deliverable_three,
                deliverable_four,
                deliverable_five,
                deliverable_six]

def test_pickle(save=True):
    if save:
        print("Pickle Saving")
        data = {
            "time" : {
                "hyper" : "This are the set of parameters",
                "running_time":[50, 52, 59, 60, 61],
                "batch_size" : [10, 20, 50 , 75, 100]
            },
            "likelihood" : {
                "hyper": "This are the set of parameters",
                "train": [1, 2, 3, 4, 5],
                "test": [0.9, 0.8, 0.7, 0.6, 0.5],
                "batch_size": [10, 20, 50 , 75, 100]
            }
        }
        pickle.dump(data, open("d3_test", "wb"))
    else:
        print("Pickle Loading")
        arr = pickle.load(open("pickle_test", "rb"))
        print(arr)


def run_deliverables(model, specific = -1):
    if specific != -1:
        DELIVERABLES[specific-1]()

    else:
        for deliverable in DELIVERABLES:
            pass



if __name__ == "__main__":
    test_pickle()
    deliverable_two("a")