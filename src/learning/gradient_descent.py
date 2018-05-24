# Standard libraries
import numpy as np


NUMITER = 20000
MINI_BATCH = 100

class GradientDescent():
    def __init__(self, hyperparameters):
        pass

    def learnParamsUsingSGD(self, model, training_corpus):
        for i in range(0, NUMITER):
            # Set gradients delta to zero.
            gradients_target = np.zeros(model.get_latent_space_size())
            gradients_context = np.zeros(model.get_latent_space_size())

            for j in (0, MINI_BATCH):
                # Sample word context pair from dataset
                target_word, context_words = training_corpus.sample_target_and_context()

                # Iterate over all context words
                for context_w in context_words:
                    # Sample K negative words
                    negative_words = model.sample_k_words()

                    # Update context and target gradients
                    context_target_softmax = model.compute_softmax(context_w, target_word)
                    gradients_context[context_w] += (1 - context_target_softmax) * model.get_target_vector(target_word)
                    gradients_target[target_word] += (1 - context_target_softmax) * model.get_context_vector(context_w)

                    for negative_w in negative_words:
                        # Update target gradient and negative context gradients
                        context_target_softmax = model.compute_softmax(negative_w, target_word)
                        gradients_context[context_w] -= (1 - context_target_softmax) * model.get_target_vector(
                            target_word)
                        gradients_target[target_word] -= (1 - context_target_softmax) * model.get_context_vector(
                            negative_w)

            # Update the model parameters according to the computed gradients
            model.update_parameters(gradients_target, gradients_context)