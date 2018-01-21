import numpy as np


def feature_derivative(errors, feature, weight, l2_penalty, feature_is_constant):

    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature)
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight

    return derivative
