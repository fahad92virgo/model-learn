import pandas as pd
import numpy as np
from mlearn.base import get_numpy_data, predict_output
from mlearn.learning_algo import gradient_descent

# Load in house sales data
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'yr_renovated': int,
              'grade': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float,
              'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


df_train_data = pd.read_csv('data/kc_house_train_data.csv', dtype=dtype_dict)
df_test_data = pd.read_csv('data/kc_house_test_data.csv', dtype=dtype_dict)

# set feature list and target variable
lst_features = ['sqft_living', 'sqft_living15']
target_variable = 'price'


# get features matrix and output vector
train_feature_matrix, train_output = get_numpy_data(df_train_data, lst_features, target_variable)


# set initial weights and learning parameters
initial_weights = np.array([-100000., 1., 1.])
step_size = 1e-12
l2_penalty = 0
max_iterations = 1000
tolerance = 1e7

# train model
weights_model = gradient_descent(train_feature_matrix, train_output, initial_weights, step_size, tolerance,
                                 max_iterations, l2_penalty)
print('feature weights: %s' % weights_model)

# evaluate mode performance
test_feature_matrix, test_output = get_numpy_data(df_test_data, lst_features, target_variable)
predictions_model = predict_output(test_feature_matrix, weights_model)
rss_model = np.sum(np.square(test_output - predictions_model))
print('Residual Sum of Squares: %s' % rss_model)
