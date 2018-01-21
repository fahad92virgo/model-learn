import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in house sales data
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'yr_renovated': int,
              'grade': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float, 'sqft_lot15': float,
              'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str, 'sqft_basement': int,
              'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('../data/kc_house_train_data.csv', dtype=dtype_dict)





