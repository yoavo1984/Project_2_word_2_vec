def predict_most_likely_context(model, input, num=1):
    """
    Predict the most likely context word with respect to the given model and the given input
    Args:
        model: The model to use to evaluate the probabilities of the possible context words.
        input: The input we want to check against.
        num: The number of input words the method should return.
        
    Returns:
        num most likely input words according to the model for the given input.
    """
    pass

def predict_most_likely_input(model, context_list, num=1):
    """
    Predicts the most likely input word with respect to the given model and the given context list.
    Args:
        model: The model to use to evaluate the probabilities of the possible input words. 
        context_list: The list we want to check against
        num: The number of context words the method should return.
        
    Returns:
        num most likely context words according to the model for the given context list.
    """
    pass

def scatter_input_in_2D(model , input_list):
    """
    Given a model and a list of input words creates a 2d scatter plot of the given words
    using the first 2 elements in each input list target vector.
    Args:
        model: The model from which to take the input list vectors for the visualization.
        input_list: A list of input words.

    Returns:
        No return value, creates a 2d pyplot scatter plot.
    """
    pass

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
    pass