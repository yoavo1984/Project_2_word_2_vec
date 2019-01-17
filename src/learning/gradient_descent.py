# Standard libraries
import numpy as np
import time
import pickle


NUMITER = 20000
DECAY_RATE = 0.5


class GradientDescent():
    def __init__(self, hyperparameters):
        self.learning_rate = hyperparameters.learning_rate
        self.batch_size = hyperparameters.batch_size
        self.decay_rate = hyperparameters.decay_iteration_num
        self.print_likelihood_iterations = hyperparameters.print_likelihood_iterations

    def create_mini_batch(self, model, training_corpus):
        batch = []
        for i in range(self.batch_size):
            target_word, context_words = training_corpus.sample_target_and_context(model.context_size)

            for context in context_words:
                negative_words = model.sample_k_words()
                batch.append((target_word, context, negative_words))
        return batch

    def learnParamsUsingSGD(self, model, training_corpus, test_corpus=False):
        print("- - Learning started.")

        np.random.seed(123)

        deliverable_data = _create_deliverable_object()
        for i in range(0, NUMITER):
            # Set gradients delta to zero.
            gradients_target = np.zeros((model.get_matrix_size()))
            gradients_context = np.zeros((model.get_matrix_size()))

            start = time.time()
            batch = self.create_mini_batch(model, training_corpus)
            for sample in batch:
                target_word, context_w, negative_words = *sample,

                # Update context and target gradients
                context_target_sigmoid = model.compute_sigmoid(context_w, target_word)
                gradients_context[context_w] += (1 - context_target_sigmoid) * model.get_target_vector(target_word)
                gradients_target[target_word] += (1 - context_target_sigmoid) * model.get_context_vector(context_w)

                for negative_w in negative_words:
                    # Update target gradient and negative context gradients
                    negative_target_sigmoid = model.compute_sigmoid(negative_w, target_word)
                    gradients_context[negative_w] -= (1/model.k) * (negative_target_sigmoid) * model.get_target_vector(
                        target_word)
                    gradients_target[target_word] -= (1/model.k) * (negative_target_sigmoid) * model.get_context_vector(
                        negative_w)

            # Multiply the gradients by the learning rate
            gradients_target *= self.learning_rate
            gradients_context *= self.learning_rate

            # Update the model parameters according to the computed gradients
            model.update_parameters(gradients_target, gradients_context)

            # Decrease the learning rate by the decay_rate
            if (i+1) % self.decay_rate == 0:
                self.learning_rate *= DECAY_RATE

            # Log the computed log likelihoods.
            if i % self.print_likelihood_iterations == 0:
                model.save_model()
                print(" -|Model Saved! iteration : {}".format(i))
                sample_likelihood = self.compute_sample_likelihood(model, training_corpus, batch, i)
                test_likelihood = self.compute_model_likelihood(model, training_corpus)
                update_deliverable(deliverable_data, sample_likelihood, test_likelihood, i)
                pickle.dump(deliverable_data, open("d1", "wb"))

        # Learning over
        print("Saving pickle ")
        pickle.dump(deliverable_data, open("d1", "wb"))

    def compute_model_likelihood(self, model, corpus):
        sum = 0
        num_of_pairs = 0
        for pair in corpus.iterate_target_context(model.context_size):

            target = pair[0]
            contexts = pair[1]

            negative = model.sample_k_words()
            for context in contexts:
                num_of_pairs += 1
                sum += model.compute_log_probability(context, target, negative)

        print("- - likelihood = {}".format(sum / num_of_pairs))
        return sum/num_of_pairs

    def compute_sample_likelihood(self, model, corpus, batch, round):
        sum = 0
        for sample in batch:
            target_word, context_w, negative_words = *sample,

            sum += model.compute_log_probability(context_w, target_word, negative_words)

        print("- | Sample likelihood = {}".format(sum / len(batch)))
        return sum / len(batch)


def _create_deliverable_object():
    data = {}
    data["iterations"] = []
    data["train"] = []
    data["test"] = []
    data["hyper"] = []
    return data


def update_deliverable(data_object, train_likelihood, test_likelihood, iteration):
    data_object["iterations"].append(iteration)
    data_object["train"].append(train_likelihood)
    data_object["test"].append(test_likelihood)
