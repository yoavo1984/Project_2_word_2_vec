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
    d1_data = pickle.load(open("d1", "rb"))

    iterations = d1_data["iterations"]
    train_error = d1_data['train']
    test_error = d1_data['test']
    hyper_string = d1_data["hyper"]


    plt.xlabel("Iteration")
    plt.ylabel("Mean Log-Likelihood")

    plt.plot(iterations, train_error)
    plt.plot(iterations, test_error)

    plt.legend(["Train Log-Likelihood" ,"Test Log-Likelihood"])
    plt.savefig("deliverable_1.png")
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

    plt.plot(d, running_time)

    # plt.show()
    plt.savefig("time_vs_d.png")
    plt.clf()

    # Plot mean log-likelihood vs d
    likelihood_data = d2_data["likelihood"]
    hyperparameters = "hyper"
    train = likelihood_data["train"]
    test = likelihood_data["test"]
    d = likelihood_data["d"]

    plt.xlabel("d")
    plt.ylabel("Mean Log-Likelihood")

    plt.plot(d, train)
    plt.plot(d, test)
    plt.legend(["Train Log-Likelihood", "Test Log-Likelihood"])

    # plt.show()
    plt.savefig("log_vs_d.png")
    plt.clf()


def deliverable_three(model):
    """
    Plot both training time and train and test (mean) log likelihood as a function of d.
    * Two separate plots.
    Args:
        model: 

    Returns:

    """
    d3_data = pickle.load(open("pickle/d3", "rb"))

    # Plot training time vs context
    time_data = d3_data["time"]
    hyperparameters = time_data["hyper"]
    running_time = time_data["running_time"]
    context_size = time_data["context_size"]

    plt.xlabel("Context Size")
    plt.ylabel("Running Time")

    plt.plot(context_size, running_time)

    # plt.show()
    plt.savefig("training_time_vs_context.png")
    plt.clf()

    # Plot mean log-likelihood vs context
    likelihood_data = d3_data["likelihood"]
    hyperparameters = likelihood_data["hyper"]
    train = likelihood_data["train"]
    test = likelihood_data["test"]
    context_size = likelihood_data["context_size"]

    plt.xlabel("Context Size")
    plt.ylabel("Mean Log-Likelihood")

    plt.plot(context_size, train)
    plt.plot(context_size, test)
    plt.legend(["Train Log-Likelihood", "Test Log-Likelihood"])

    # plt.show()
    plt.savefig("likelihood_vs_context.png")
    plt.clf()


def deliverable_four(model):
    given_words = ["good", "bad", "lame", "cool", "exciting"]
    for word in given_words:
        # Find 10 most likely contexts
        print("Context for :{}".format(word))
        matching_contexts = predict_most_likely_context(model, word, 10)

    given_words.extend(["john", "alice", "computer", "learning", "machine"])
    scatter_input_in_2d(model, given_words)
    scatter_input_in_2d(model, given_words, False)


def deliverable_five(model):
    print("The movie was surprisingly ______")
    predict_most_likely_input(model, "the movie was surprisingly".split(), 4)
    print("*"*50)
    print("____ was really disappointing")
    predict_most_likely_input(model, "was really disappointing".split(), 0)
    print("*" * 50)
    print("Knowing that she ____ was the best part")
    predict_most_likely_input(model, "knowing that she was the best part".split(), 3)


def deliverable_six(model):
    print("man is to woman as men is to ____")
    analogy_solver(model, "man", "woman", "men")

    print("good is to great as bad is to ____")
    analogy_solver(model, "good", "great", "bad")

    print("warm is to cold as summer is to ____")
    analogy_solver(model, "warm", "cold", "summer")
    pass


DELIVERABLES = [deliverable_one,
                deliverable_two,
                deliverable_three,
                deliverable_four,
                deliverable_five,
                deliverable_six]


def run_deliverables(model, specific = -1):
    if specific != -1:
        DELIVERABLES[specific-1](model)

    else:
        for deliverable in DELIVERABLES:
            pass



if __name__ == "__main__":
    # deliverable_one("a")
    # deliverable_two("a")
    deliverable_one("a")
