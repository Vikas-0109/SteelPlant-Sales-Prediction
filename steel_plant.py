import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_excel('sales_customers.xls')

# Clean column names
data.columns = data.columns.str.strip()

# Print the first few rows and column names to check the data
st.write("Data preview:")
st.write(data.head())
st.write("Column names:")
st.write(data.columns)

# Check for the exact name of the 'YEAR_MONTH' column
if 'YEAR_MONTH' in data.columns:
    date_col = 'YEAR_MONTH'
else:
    st.error("Date column 'YEAR_MONTH' not found in the dataset.")
    raise KeyError("The required date column 'YEAR_MONTH' is not found.")

# Convert the date column to datetime
data[date_col] = pd.to_datetime(data[date_col])

# Set the date column as the index
data.set_index(date_col, inplace=True)

# Create lag features
data['lag_1'] = data['SUM(SALES_VALUE_TOT)'].shift(1)
data['lag_2'] = data['SUM(SALES_VALUE_TOT)'].shift(2)
data.dropna(inplace=True)

# Split the data into training and testing sets
X = data[['lag_1', 'lag_2']]
y = data['SUM(SALES_VALUE_TOT)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Function to predict future sales for a given date
def predict_sales(input_date):
    input_date = pd.to_datetime(input_date, format='%Y-%m')

    # Generate a date range up to the input date
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), end=input_date, freq='MS')

    # Create a DataFrame for future predictions
    future_data = pd.DataFrame(index=future_dates)
    future_data['lag_1'] = np.nan
    future_data['lag_2'] = np.nan

    # Fill the lag features based on the latest available data
    last_known_value = data.iloc[-1]['SUM(SALES_VALUE_TOT)']
    if len(future_data) > 0:
        future_data.iloc[0, future_data.columns.get_loc('lag_1')] = last_known_value
        if len(future_data) > 1:
            future_data.iloc[1, future_data.columns.get_loc('lag_2')] = last_known_value

        for i in range(1, len(future_data)):
            future_data.iloc[i, future_data.columns.get_loc('lag_1')] = future_data.iloc[i-1, future_data.columns.get_loc('predicted_sales')] if 'predicted_sales' in future_data.columns else future_data.iloc[i-1, future_data.columns.get_loc('lag_1')]
            if i > 1:
                future_data.iloc[i, future_data.columns.get_loc('lag_2')] = future_data.iloc[i-2, future_data.columns.get_loc('predicted_sales')] if 'predicted_sales' in future_data.columns else future_data.iloc[i-2, future_data.columns.get_loc('lag_2')]

        # Predict future sales
        future_data['predicted_sales'] = model.predict(future_data[['lag_1', 'lag_2']])

        # If input_date is not in the index, return the closest future prediction
        if input_date not in future_data.index:
            st.warning(f"The input date {input_date} is not in the index. Returning the closest future prediction.")
            return future_data.iloc[-1]['predicted_sales']
        else:
            return future_data.loc[input_date, 'predicted_sales']
    else:
        st.error("Input date is before the last known date in the dataset.")
        return None

# Streamlit interface
st.title("Sales Prediction Application")

user_input_date = st.text_input("Enter a date (YYYY-MM):", "2024-03")
if user_input_date:
    predicted_sales = predict_sales(user_input_date)
    st.write(f"Predicted sales for {user_input_date}: {predicted_sales}")