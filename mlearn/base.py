import numpy as np
import pandas as pd
from math import sqrt


# --------------------------------------------------------------------------------------------------------------------
# Base functions
# --------------------------------------------------------------------------------------------------------------------
def get_numpy_data(dataframe, features, output):

    """
    Parameters
    ----------
    :param dataframe: pandas dataframe containing the features and target variable
    :param features: list of features containing the names of the columns to be included as features
    :param output: name of the target variable as a string


    Returns
    ---------
    :return: feature_matrix: numpy matrix containing the features as columns
    :return: output_array: numpy array with the target variable values
    """

    # add a constant column to a Pandas Data Frame
    dataframe['constant'] = 1

    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features

    # select the columns of dataFrame given by the features list:
    features_dataframe = pd.DataFrame(dataframe, columns=features)

    # convert the features_dataFrame into a numpy matrix:
    feature_matrix = features_dataframe.as_matrix()

    # assign the column of dataframe associated with the output to a numpy array
    output_array = dataframe[output].values

    return feature_matrix, output_array


def predict_output(feature_matrix, weights):
    """
    Parameters
    ----------
    :param feature_matrix: numpy matrix containing the features as columns
    :param weights: corresponding numpy array containing the weight of each feature

    :return: numpy array containig the predictions
    """

    predictions = np.dot(feature_matrix, weights)

    return predictions


# --------------------------------------------------------------------------------------------------------------------
# Linear Regression Implementation
# --------------------------------------------------------------------------------------------------------------------
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2 * np.dot(errors, feature)
    return derivative


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, max_iter):
    converged = False
    weights = np.array(initial_weights) # make sure it's a numpy array
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
            derivative = feature_derivative(errors, feature_matrix[:, i])

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


# --------------------------------------------------------------------------------------------------------------------
# Ridge Regression Implementation
# --------------------------------------------------------------------------------------------------------------------
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature)
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array

    # while not reached maximum number of iterations:
    for n in range(max_iterations):

        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        for i in range(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, False)
            # subtract the step size times the derivative from the current weight

            weights[i] = weights[i] - (step_size * derivative)

    return weights
