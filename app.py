from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import sys
import sklearn

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# --- Configuration ---
DATA_PATH = "weatherAUS.csv"
MODELS_DIR = "models"
TEMP_REG_MODEL_PATH = os.path.join(MODELS_DIR, "avgtemp_reg_compressed.pkl")
RAIN_CLF_MODEL_PATH = os.path.join(MODELS_DIR, "rain_today_clf_compressed.pkl")
LOC_ENC_MODEL_PATH = os.path.join(MODELS_DIR, "loc_encoder_compressed.pkl")

# Global variables to store loaded data and models
df = None
clf = None
reg = None
le = None
unique_locations = []

def load_data():
    global df
    try:
        if not os.path.exists(DATA_PATH):
            print(f"❌ Dataset file NOT FOUND: {DATA_PATH}")
            return False
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def load_models():
    global clf, reg, le
    print("--- Flask Environment Info ---")
    print(f"Python Version: {sys.version}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    print(f"Joblib Version: {joblib.__version__}")
    print("-------------------------------------")

    try:
        clf = joblib.load(RAIN_CLF_MODEL_PATH)
        reg = joblib.load(TEMP_REG_MODEL_PATH)
        le = joblib.load(LOC_ENC_MODEL_PATH)
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ General error during model loading: {e}")
        return False

# Initialize data and models when app starts
def initialize_app():
    global unique_locations
    if not load_data():
        return False
    if df is None or df.empty:
        print("The loaded dataset is empty. Cannot proceed.")
        return False
    if not load_models():
        print("❌ Failed to load one or more machine learning models.")
        return False
    
    unique_locations = sorted(df["Location"].unique())
    return True

@app.route('/')
def index():
    if not initialize_app():
        flash("❌ Failed to initialize the application. Please check data and model files.", "error")
        return render_template('error.html')
    
    # Default date
    default_date = datetime(2026, 5, 22).strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                         locations=unique_locations, 
                         default_date=default_date)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_location = request.form.get('location')
        selected_date_str = request.form.get('date')
        
        if not selected_location or not selected_date_str:
            flash("Please select both a location and a date.", "warning")
            return render_template('index.html', 
                                 locations=unique_locations, 
                                 default_date=selected_date_str or datetime.now().strftime('%Y-%m-%d'))
        
        # Parse the date
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        
        # Prepare input features
        input_data = pd.DataFrame({
            'Location_Encoded': [le.transform([selected_location])[0]],
            'MinTemp': [df['MinTemp'].mean()],
            'MaxTemp': [df['MaxTemp'].mean()],
            'Humidity9am': [df['Humidity9am'].mean()],
            'Pressure9am': [df['Pressure9am'].mean()],
            'WindSpeed9am': [df['WindSpeed9am'].mean()],
            'Year': [selected_date.year],
            'Month': [selected_date.month],
            'Day': [selected_date.day]
        })

        # Ensure columns are in the same order as training data
        feature_cols = ["Location_Encoded", "MinTemp", "MaxTemp", "Humidity9am",
                       "Pressure9am", "WindSpeed9am", "Year", "Month", "Day"]
        input_data = input_data[feature_cols]

        # Make predictions
        predicted_avg_temp = reg.predict(input_data)[0]
        rain_today_prediction = clf.predict(input_data)[0]
        rain_today_label = "Yes" if rain_today_prediction == 1 else "No"

        result = {
            'location': selected_location,
            'date': selected_date.strftime('%Y-%m-%d'),
            'rain_today': rain_today_label,
            'avg_temp': round(predicted_avg_temp, 2)
        }

        return render_template('result.html', result=result)
        
    except Exception as e:
        flash(f"❌ Error during prediction: {str(e)}", "error")
        return render_template('index.html', 
                             locations=unique_locations, 
                             default_date=datetime.now().strftime('%Y-%m-%d'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON responses"""
    try:
        data = request.get_json()
        selected_location = data.get('location')
        selected_date_str = data.get('date')
        
        if not selected_location or not selected_date_str:
            return jsonify({'error': 'Missing location or date'}), 400
        
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        
        # Prepare input features
        input_data = pd.DataFrame({
            'Location_Encoded': [le.transform([selected_location])[0]],
            'MinTemp': [df['MinTemp'].mean()],
            'MaxTemp': [df['MaxTemp'].mean()],
            'Humidity9am': [df['Humidity9am'].mean()],
            'Pressure9am': [df['Pressure9am'].mean()],
            'WindSpeed9am': [df['WindSpeed9am'].mean()],
            'Year': [selected_date.year],
            'Month': [selected_date.month],
            'Day': [selected_date.day]
        })

        feature_cols = ["Location_Encoded", "MinTemp", "MaxTemp", "Humidity9am",
                       "Pressure9am", "WindSpeed9am", "Year", "Month", "Day"]
        input_data = input_data[feature_cols]

        # Make predictions
        predicted_avg_temp = reg.predict(input_data)[0]
        rain_today_prediction = clf.predict(input_data)[0]
        rain_today_label = "Yes" if rain_today_prediction == 1 else "No"

        return jsonify({
            'location': selected_location,
            'date': selected_date.strftime('%Y-%m-%d'),
            'rain_today': rain_today_label,
            'avg_temp': round(predicted_avg_temp, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
