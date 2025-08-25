from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌦️ Weather Prediction App</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .status { background: #f0f8ff; padding: 20px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌦️ Weather Prediction App</h1>
            <div class="status">
                <h3>App Status:</h3>
                <p>✅ Flask app is running successfully!</p>
                <p>📁 Data file exists: """ + str(os.path.exists("weatherAUS.csv")) + """</p>
                <p>🤖 Models folder exists: """ + str(os.path.exists("models")) + """</p>
                <p><a href="/health">Check detailed health →</a></p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health():
    import pandas as pd
    import joblib
    
    status = {
        'app_running': True,
        'files_exist': {
            'data': os.path.exists('weatherAUS.csv'),
            'models_dir': os.path.exists('models')
        },
        'working_directory': os.getcwd(),
        'files_in_root': os.listdir('.') if os.path.exists('.') else []
    }
    
    # Try to load models
    try:
        models_dir = "models"
        model_files = [
            os.path.join(models_dir, "avgtemp_reg_compressed.pkl"),
            os.path.join(models_dir, "rain_today_clf_compressed.pkl"),
            os.path.join(models_dir, "loc_encoder_compressed.pkl")
        ]
        
        models_exist = all(os.path.exists(f) for f in model_files)
        status['models_exist'] = models_exist
        
        if models_exist:
            # Try loading one model
            clf = joblib.load(model_files[1])
            status['model_loading'] = 'SUCCESS'
        else:
            status['model_loading'] = 'MODELS_NOT_FOUND'
            
    except Exception as e:
        status['model_loading'] = f'ERROR: {str(e)}'
    
    # Return as HTML for easy viewing
    html = "<h1>App Health Status</h1><pre>" + str(status) + "</pre>"
    html += "<p><a href='/'>← Back to main page</a></p>"
    return html

if __name__ == '__main__':
    app.run()
