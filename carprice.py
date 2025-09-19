import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
MODEL_PATH = "rf_model.joblib"
SCALER_PATH = "scaler.joblib"
ENCODERS_PATH = "encoders.joblib"
@st.cache_data
def load_data():
	url="https://raw.githubusercontent.com/greeshmavilasagarapu/car-price-prediction/main/car_sales_data.csv"
	return pd.read_csv(url)
orig_df = load_data()
st.title("🚗 Car Price Prediction App")
st.write("This app predicts the **selling price of a car** using Machine Learning.")

if st.checkbox("Show raw dataset"):
	st.write(orig_df.head())
df = orig_df.copy()
df = df.dropna(subset=['Price']).copy()

# Handle categorical and numeric column processing
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
categorical_options = {col: sorted(df[col].dropna().unique().astype(str)) for col in categorical_cols}

encoders = {}
for col in categorical_cols:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col].astype(str))
	encoders[col] = le
if 'Year' in df.columns:
	df['Car_Age'] = 2025 - df['Year']
	df.drop('Year', axis=1, inplace=True)
X = df.drop('Price', axis=1)
y = df['Price']

# --- After you define X and y ---
@st.cache_resource
def load_model_and_preprocessors(df):
	categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
	encoders = {}
	for col in categorical_cols:
		le = LabelEncoder()
		df[col] = le.fit_transform(df[col].astype(str))
		encoders[col] = le
	if 'Year' in df.columns:
		df['Car_Age'] = 2025 - df['Year']
		df.drop('Year', axis=1, inplace=True)
	X = df.drop('Price', axis=1)
	y = df['Price']
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	model = RandomForestRegressor()
	model.fit(X_scaled, y)
	return model, scaler, encoders, X, categorical_cols

# Load dataset once
orig_df = load_data()
df = orig_df.dropna(subset=['Price']).copy()

# Train model once per session (cached)
model, scaler, encoders, X, categorical_cols = load_model_and_preprocessors(df)


# Display metrics
y_pred = model.predict(X_scaled)
#st.write(f"R²: {r2_score(y, y_pred)}")
#st.write(f"MAE: {mean_absolute_error(y, y_pred)}")
#st.write(f"MSE: {mean_squared_error(y, y_pred)}")
inputs = {}
for col in X.columns:
	if col in categorical_cols:
		options = categorical_options[col]
		default = options[0]
		inputs[col] = st.selectbox(col, options, index=options.index(default))
	else:
		default = float(orig_df[col].median())
		inputs[col] = st.number_input(col, value=default, min_value=0.0)

if st.button("Predict"):
	input_df = pd.DataFrame([inputs])
	for col in categorical_cols:
		if col in input_df.columns:
			input_df[col] = encoders[col].transform(input_df[col].astype(str))
	input_scaled = scaler.transform(input_df)
	prediction = model.predict(input_scaled)
	st.success(f"💰 Estimated Car Price: ₹ {prediction[0]:,.2f}")




