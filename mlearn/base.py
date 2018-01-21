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
    :return feature_matrix: numpy matrix containing the features as columns
    :return output_array: numpy array with the target variable values
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

    Returns
    ---------
    :return numpy array containing the predictions
    """

    predictions = np.dot(feature_matrix, weights)

    return predictions


def normalize_features(feature_matrix):
    """
    Parameters
    ----------
    :param feature_matrix: numpy matrix containing the features as columns

    :return normalized_features: numpy array containing the normalized featured as columns
    :return norms: numpy array containing the 2-norm of the columns in the feature_matrix
    """
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms

    return normalized_features, norms
