class LearningHyperparameters():
    def __init__(self, learning_rate, batch_size, decay_iteration_num, print_likelihood_iterations=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_iteration_num = decay_iteration_num
        self.print_likelihood_iterations = print_likelihood_iterations