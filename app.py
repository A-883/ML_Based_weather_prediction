
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Ensure numpy is imported
from datetime import datetime
import sys
import sklearn

# --- Configuration ---
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl")

# --- Load dataset ---
@st.cache_data(show_spinner="Loading weather data...")
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Dataset file NOT FOUND: {DATA_PATH}")
            return None
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return df
    except Exception as e:
        st.exception(f"❌ Error loading dataset: {e}")
        return None

# --- Load models safely ---
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    st.info("--- Streamlit Cloud Environment Info ---")
    st.info(f"Python Version: {sys.version}")
    st.info(f"Pandas Version: {pd.__version__}")
    st.info(f"Scikit-learn Version: {sklearn.__version__}")
    st.info(f"Joblib Version: {joblib.__version__}")
    st.info("-------------------------------------") # End debug info

    try:
        # No need for individual file found messages in production, assuming they load
        temp_reg = joblib.load(TEMP_REG_MODEL_PATH)
        rain_clf = joblib.load(RAIN_CLF_MODEL_PATH)
        loc_enc = joblib.load(LOC_ENC_MODEL_PATH)
        st.success("✅ All models loaded successfully!")
        return rain_clf, temp_reg, loc_enc
    except Exception as e:
        st.exception(f"❌ General error during model loading: {e}")
        return None, None, None


# --- Main App Logic ---

df = load_data()
if df is None:
    st.error("❌ Failed to load the dataset. Please check `weatherAUS.csv` file.")
    st.stop() # Stops the app if data load fails

if df.empty:
    st.error("The loaded dataset is empty. Cannot proceed.")
    st.stop() # Stops the app if dataset is empty

clf, reg, le = load_models()
if clf is None or reg is None or le is None:
    st.error("❌ Failed to load one or more machine learning models. Please check the model files and dependencies.")
    st.stop() # Stops the app if model load fails

st.title("🌦️ Weather Prediction App")
st.markdown("Enter the desired location and date to get the weather prediction.")


# --- User Input Widgets ---
unique_locations = sorted(df["Location"].unique())
selected_location = st.selectbox("Select Location:", unique_locations)

# Default date to today, or a sensible date if data is old
today = datetime.now().date()
default_date = datetime(2026, 5, 22).date() # Or a date within your training data range for better prediction
selected_date = st.date_input("Select Date:", value=default_date)


if st.button("Predict Weather"):
    if selected_location and selected_date:
        # Prepare input features
        input_data = pd.DataFrame({
            'Location_Encoded': [le.transform([selected_location])[0]],
            'MinTemp': [df['MinTemp'].mean()], # Use mean of training data for missing inputs
            'MaxTemp': [df['MaxTemp'].mean()],
            'Humidity9am': [df['Humidity9am'].mean()],
            'Pressure9am': [df['Pressure9am'].mean()],
            'WindSpeed9am': [df['WindSpeed9am'].mean()],
            'Year': [selected_date.year],
            'Month': [selected_date.month],
            'Day': [selected_date.day]
        })

        # Ensure columns are in the same order as training data
        # Get feature names from the model's training (X_temp and X_rain from train_model.py)
        # Assuming you used these features:
        feature_cols = ["Location_Encoded", "MinTemp", "MaxTemp", "Humidity9am",
                        "Pressure9am", "WindSpeed9am", "Year", "Month", "Day"]
        input_data = input_data[feature_cols]

        # Make predictions
        predicted_avg_temp = reg.predict(input_data)[0]
        rain_today_prediction = clf.predict(input_data)[0]
        rain_today_label = "Yes" if rain_today_prediction == 1 else "No"

        st.subheader("--- Prediction Results ---")
        st.write(f"**Location:** {selected_location}, **Date:** {selected_date.strftime('%Y-%m-%d')}")
        st.write(f"**Rain Today:** {rain_today_label}")
        st.write(f"**Predicted Average Temperature:** {predicted_avg_temp:.2f} °C")
    else:
        st.warning("Please select both a location and a date.")
