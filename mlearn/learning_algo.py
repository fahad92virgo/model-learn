import numpy as np
from .base import predict_output
from .derivatives import feature_derivative
from math import sqrt


def gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, max_iter=1000, l2_penalty=0):

    converged = False
    weights = np.array(initial_weights)
    iter_count = 0

    while not converged:

        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = predictions - output

        # initialize the gradient sum of squares
        gradient_sum_squares = 0

        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)):
            # feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:

            if i == 0:
                derivative = feature_derivative(errors, feature_matrix[:, i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative(errors, feature_matrix[:, i], weights[i], l2_penalty, False)

            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += derivative ** 2

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (step_size * derivative)

        iter_count += 1

        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)

        print('iter %s | gradient_magnitude: %s' % (iter_count, gradient_magnitude))

        if gradient_magnitude < tolerance or iter_count >= max_iter:
            converged = True

    return weights
