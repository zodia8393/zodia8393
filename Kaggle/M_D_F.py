import pandas as pd
import numpy as np

# Load the train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
census_data = pd.read_csv('census_starter.csv')

# Merge the train data with the census data
train_data = pd.merge(train_data, census_data, on='cfips')
test_data = pd.merge(test_data, census_data, on='cfips')

# Drop the columns that are not useful for the model
train_data = train_data.drop(['row_id', 'county_name', 'state_name'], axis=1)
test_data = test_data.drop(['row_id', 'county_name', 'state_name'], axis=1)

# Convert the first_day_of_month column to datetime
train_data['first_day_of_month'] = pd.to_datetime(train_data['first_day_of_month'])
test_data['first_day_of_month'] = pd.to_datetime(test_data['first_day_of_month'])

# Extract features from the first_day_of_month column
train_data['year'] = train_data['first_day_of_month'].dt.year
train_data['month'] = train_data['first_day_of_month'].dt.month
test_data['year'] = test_data['first_day_of_month'].dt.year
test_data['month'] = test_data['first_day_of_month'].dt.month

# Drop the first_day_of_month column
train_data = train_data.drop(['first_day_of_month'], axis=1)
test_data = test_data.drop(['first_day_of_month'], axis=1)

# Split the train data into features and target
X_train = train_data.drop(['microbusiness_density'], axis=1)
y_train = train_data['microbusiness_density']

# Fill missing values in the test data with the mean
test_data.fillna(np.mean(train_data), inplace=True)

# Scale the features in the train and test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(test_data)

# load the train and test data into dataframes
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# merge the train and test data into a single dataframe
merged_df = pd.concat([train_df, test_df], sort=False)

# convert the first_day_of_month column to a datetime
merged_df['first_day_of_month'] = pd.to_datetime(merged_df['first_day_of_month'])

# extract the year and month from the datetime
merged_df['year'] = merged_df['first_day_of_month'].dt.year
merged_df['month'] = merged_df['first_day_of_month'].dt.month

# one hot encode the state_name column
state_dummies = pd.get_dummies(merged_df['state_name'], prefix='state')
merged_df = pd.concat([merged_df, state_dummies], axis=1)

# separate the merged dataframe into train and test dataframes
train_df = merged_df[merged_df['microbusiness_density'].notnull()]
test_df = merged_df[merged_df['microbusiness_density'].isnull()].drop(['microbusiness_density'], axis=1)

# drop the original state_name column and the first_day_of_month column
train_df.drop(['state_name', 'first_day_of_month'], axis=1, inplace=True)
test_df.drop(['state_name', 'first_day_of_month'], axis=1, inplace=True)

# Create new features from existing features

# Create a column for the number of businesses per capita
train_data['business_per_capita'] = train_data['business_count'] / train_data['population']
test_data['business_per_capita'] = test_data['business_count'] / test_data['population']

# Create a column for the number of businesses per square mile
train_data['business_per_sq_mile'] = train_data['business_count'] / train_data['land_area']
test_data['business_per_sq_mile'] = test_data['business_count'] / test_data['land_area']

# Create a column for the population density
train_data['population_density'] = train_data['population'] / train_data['land_area']
test_data['population_density'] = test_data['population'] / test_data['land_area']

# Create a column for the ratio of businesses to population
train_data['business_to_population_ratio'] = train_data['business_count'] / train_data['population']
test_data['business_to_population_ratio'] = test_data['business_count'] / test_data['population']

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Split the data into features (X) and target (y)
X = train_data.drop(['microbusiness_density'], axis=1)
y = train_data['microbusiness_density']

# Initialize the models
rf_model = RandomForestRegressor()
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor()

# Cross-validate the models using 10-fold cross-validation
kf = KFold(n_splits=10, random_state=42)
rf_scores = cross_val_score(rf_model, X, y, cv=kf)
lr_scores = cross_val_score(lr_model, X, y, cv=kf)
dt_scores = cross_val_score(dt_model, X, y, cv=kf)

# Print the average performance of each model
print("Random Forest Regressor Performance: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
print("Linear Regression Performance: %0.2f (+/- %0.2f)" % (lr_scores.mean(), lr_scores.std() * 2))
print("Decision Tree Regressor Performance: %0.2f (+/- %0.2f)" % (dt_scores.mean(), dt_scores.std() * 2))

# Import the necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Train a random forest regression model on the training data
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the performance of the model
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error: ", mae)

# Load the test data
X_test = test_data.drop(['microbusiness_density'], axis=1)
y_test = test_data['microbusiness_density']

# Make predictions on the test data using the trained model
y_pred = model.predict(X_test)

# Evaluate the performance of the model using mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Evaluate the performance of the model using mean squared error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Evaluate the performance of the model using root mean squared error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Evaluate the performance of the model using R-squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)


# Importing required libraries
import pandas as pd
from sklearn.externals import joblib

# Loading the test dataset
test_data = pd.read_csv('test.csv')

# Preprocessing the test data in the same way as the train data
test_data = preprocess_data(test_data)

# Loading the trained model from disk
model = joblib.load('model.pkl')

# Making predictions on the test data
predictions = model.predict(test_data)

# Saving the predictions to a CSV file
pd.DataFrame({'Id': test_data.index, 'Target': predictions}).to_csv('predictions.csv', index=False)

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
}

# Create the base model
base_model = RandomForestRegressor(random_state=0)

# Use GridSearchCV to fine-tune the model
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Use the best parameters to train the final model
final_model = RandomForestRegressor(**grid_search.best_params_)
final_model.fit(X_train, y_train)

# Final Submission Part

# Load the test data
test = pd.read_csv("test.csv")

# Preprocess the test data
test_processed = preprocess_data(test)

# Make predictions on the test data
predictions = model.predict(test_processed)

# Prepare the submission file
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)


