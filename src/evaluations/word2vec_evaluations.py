import numpy as np
import matplotlib.pyplot as plt

C = 3

def predict_most_likely_context(model, input, num=1):
    """
    Predict the most likely context word with respect to the given model and the given input
    Args:
        model: The model to use to evaluate the probabilities of the possible context words.
        input: The input we want to check against(a target word).
        num: The number of input words the method should return.
        
    Returns:
        num most likely input words according to the model for the given input.
    """
    target_vector = model.get_target_vector(input)
    answers_index = model.find_most_similar_vector(target_vector, True)
    for index in answers_index:
        answer = model.get_word_by_index(index)
        print(answer)


def predict_most_likely_input(model, context_list, input_position, num=1):
    """
    Predicts the most likely input word with respect to the given model and the given context list.
    Args:
        model: The model to use to evaluate the probabilities of the possible input words. 
        context_list: The list we want to check against
        input_position: The position of the input word.
        num: The number of input words the method should return.
        
    Returns:
        num most likely context words according to the model for the given context list.
    """

    # Get context size from the model.
    context_size = model.context_size

    # Pull context words from the given context_list
    contexts = []
    for i in range(context_size):
        if i == 0:
            if input_position < len(context_list):
                contexts.append(context_list[input_position])
        else:
            if input_position - i >= 0:
                contexts.append(context_list[input_position - i])
            if input_position + i < (len(context_list) - 1):
                contexts.append(context_list[input_position + i])

    # Compute likelihood for each
    num_of_words = model.get_number_of_words()
    words_scores = np.zeros(num_of_words)
    for word in range(num_of_words):
        word_score = 0
        for context in contexts:
            negatives = model.sample_k_words()
            context_id = model.get_index_by_word(context)
            word_score += model.compute_log_probability(context_id, word, negatives)

        words_scores[word] = word_score

    best_inputs_indexes = np.argpartition(words_scores, -10)[-10:]
    for word_index in best_inputs_indexes:
        print(model.get_word_by_index(word_index))


def scatter_input_in_2d(model , input_list, target=True):
    """
    Given a model and a list of input words creates a 2d scatter plot of the given words
    using the first 2 elements in each input list target vector.
    Args:
        model: The model from which to take the input list vectors for the visualization.
        input_list: A list of input words.

    Returns:
        No return value, creates a 2d pyplot scatter plot.
    """
    to_plot_x = []
    to_plot_y = []

    # Pull all the words vector from the model.
    for word in input_list:
        if target:
            target_vector = model.get_target_vector(word)
            to_plot_x.append(target_vector[0])
            to_plot_y.append(target_vector[1])
        else:
            context_vector = model.get_context_vector(word)
            to_plot_x.append(context_vector[0])
            to_plot_y.append(context_vector[1])

    # Create the scatter plot.
    fig, ax = plt.subplots()
    ax.scatter(to_plot_x, to_plot_y)

    # Annotate each word in the plot.
    for i, word in enumerate(input_list):
        ax.annotate(word, (to_plot_x[i], to_plot_y[i]))

    plt.title("Word Embedding Visualization")
    # plt.scatter(to_plot_x, to_plot_y)
    plt.show()
    plt.savefig("scatter_plot")


def analogy_solver(model, a, b, c):
    """
    Solve the following analogy given the model : a to b is as c to d where
    d is the word we are trying to find.
    This is done by vector arithmetic :  argmax d for which d(a - b + c) is maximal.
    Args:
        model: 
        a: 
        b: 
        c: 

    Returns:
        A word d for which argmax d for which d(a - b + c) is maximal.
    """

    # Get vectors from model.
    a_vector = model.get_target_vector(a)
    b_vector = model.get_target_vector(b)
    c_vector = model.get_target_vector(c)

    # Do the vector math (a - b + c)
    find = a_vector - b_vector + c_vector
    find_norm = np.linalg.norm(find)
    find = find / find_norm

    # Ask model to find most similar target.
    answers_index = model.find_most_similar_vector(find)
    for index in answers_index:
        answer = model.get_word_by_index(index)
        print(answer)

