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
	return pd.read_csv(r"C:\Users\Admin\Downloads\car_sales_data.csv")
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

# Scaling numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODERS_PATH):
	model = joblib.load(MODEL_PATH)
	scaler = joblib.load(SCALER_PATH)
	encoders = joblib.load(ENCODERS_PATH)
	model_trained = True
else:
	model = RandomForestRegressor()
	model.fit(X_scaled, y)
	joblib.dump(model, MODEL_PATH)
	joblib.dump(scaler, SCALER_PATH)
	joblib.dump(encoders, ENCODERS_PATH)
	model_trained = True

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
