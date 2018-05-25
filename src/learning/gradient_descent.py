# Standard libraries
import numpy as np


NUMITER = 500
DECAY_RATE = 0.5

class GradientDescent():
    def __init__(self, hyperparameters):
        self.learning_rate = hyperparameters.learning_rate
        self.batch_size = hyperparameters.batch_size
        self.decay_rate = hyperparameters.decay_iteration_num

    def create_mini_batch(self, model, training_corpus):
        batch = []
        for i in range(self.batch_size):
            target_word, context_words = training_corpus.sample_target_and_context(model.context_size)
            for context in context_words:
                negative_words = model.sample_k_words()
                batch.append((target_word, context, negative_words))

        return batch

    def learnParamsUsingSGD(self, model, training_corpus):
        print("- - Learning started.")
        for i in range(0, NUMITER):
            # Set gradients delta to zero.
            gradients_target = np.zeros((model.get_matrix_size()))
            gradients_context = np.zeros((model.get_matrix_size()))

            batch = self.create_mini_batch(model, training_corpus)
            print ("- = Before")
            self.compute_sample_likelihood(model, training_corpus, batch)
            for sample in batch:
                target_word, context_w, negative_words = *sample,

                # Update context and target gradients
                context_target_softmax = model.compute_softmax(context_w, target_word)
                gradients_context[context_w] += (1 - context_target_softmax) * model.get_target_vector(target_word)
                gradients_target[target_word] += (1 - context_target_softmax) * model.get_context_vector(context_w)

                for negative_w in negative_words:
                    # Update target gradient and negative context gradients
                    context_target_softmax = model.compute_softmax(negative_w, target_word)
                    gradients_context[context_w] -= (context_target_softmax) * model.get_target_vector(
                        target_word)
                    gradients_target[target_word] -= (context_target_softmax) * model.get_context_vector(
                        negative_w)

            # Multiply the gradients by the learning rate
            gradients_target *= self.learning_rate
            gradients_context *= self.learning_rate

            # Update the model parameters according to the computed gradients
            model.update_parameters(gradients_target, gradients_context)

            # Decrease the learning rate by the decay_rate
            if i == self.decay_rate:
                self.learning_rate /= DECAY_RATE

            print("- = After")
            self.compute_sample_likelihood(model, training_corpus, batch)

    def compute_model_likelihood(self, model, corpus, round):
        sum = 0
        print("{}.".format(round))
        for pair in corpus.iterate_target_context(model.context_size):
            target = pair[0]
            contexts = pair[1]
            negative = model.sample_k_words()

            for context in contexts:
                sum += model.compute_log_probability(context, target, negative)

        print ("- - likelihood = {}".format(sum))


    def compute_sample_likelihood(self, model, corpus, batch):
        sum = 0
        for sample in batch:
            target_word, context_w, negative_words = *sample,

            sum += model.compute_log_probability(context_w, target_word, negative_words)

        print("- | Sample likelihood = {}".format(sum))