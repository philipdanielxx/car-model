import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from logging import StreamHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
cars = pd.read_csv("C:\\Users\\chuks\\Downloads\\archive (1)\\cars_2010_2020.csv")

# Assuming the correct column is "Price" for the target variable
cars["Price"] = cars["Price (USD)"]

# Title of the app
st.title("Cars price predictiion")

# Data Overview
st.header("data overview for first 10 row")
st.write(cars.head(10))

# Encoding non-numeric columns

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cars['Make'] = encoder.fit_transform(cars['Make'])  # Encoding for 'Make'
cars['Model'] = encoder.fit_transform(cars['Model'])  # Encoding for 'Model'
cars['Fuel Type'] = encoder.fit_transform(cars['Fuel Type'])  # Encoding for 'Fuel Type'


# split the data into input and output
X = cars.drop('Price (USD)', axis=1) # input features
y = cars['Price (USD)'] # target
X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
sel_mod = st.selectbox("select a model", ["Linear Regression", "Random Forest", "Decision Tree"])

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(),
          "Decision Tree": DecisionTreeRegressor()}

# Train the model
selected = models[sel_mod] # initialises the model

# train the selected model
selected.fit(X_train, y_train)

# make predictions
y_pred = selected.predict(X_test)

# model evaluation
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

# display the units
st.write(f"R2score: {R2}")
st.write(f"Mean_Squared_Error: {MSE}")
st.write(f"mean_Absolute_Error: {MAE}")

# prompt for user input
st.write("enter the input values for prediction:")

user_input = {}
for column in X.columns:
    min_val = float(np.min(X[column]))  # Convert to float
    max_val = float(np.max(X[column]))  # Convert to float
    mean_val = float(np.mean(X[column]))  # Convert to float
    
    # Ensure all values are of the same type (float)
    user_input[column] = st.number_input(
        column,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Convert dictionary to dataframe
user_input_df = pd.DataFrame([user_input])

# Standardize the dataframe
user_input_sc_df = scaler.transform(user_input_df)

# Make predictions for the price
price_predicted = selected.predict(user_input_sc_df)

# Display the predicted price
st.write(f"Predicted price is: {price_predicted[0] * 100000}")

# user_input = {}
# for column in X.columns:
#    user_input[column] = st.number_input(column, min_value = np.min(X[column]), max_value = np.max(X[column]), value = np.mean(X[column]))
    
# convert dictionary to dataframe
#user_input_df = pd.DataFrame([user_input])

# standardise the dataframe
#user_input_sc_df = scaler.transform(user_input_df)

# make predictions for the price
#price_predicted = selected.predict(user_input_sc_df)

# display the predicted price
#st.write(f"predicted price is: {price_predicted[0] * 100000}")
