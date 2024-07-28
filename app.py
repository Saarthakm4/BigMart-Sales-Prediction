from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the scaler and model during app initialization
scaler_path = os.path.join('models', 'sc.sav')
model_path = os.path.join('models', 'lr.sav')

if os.path.exists(scaler_path):
    sc = joblib.load(scaler_path)
else:
    raise FileNotFoundError(f"Scaler file not found at path: {scaler_path}")

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at path: {model_path}")

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            required_fields = ['item_weight', 'item_fat_content', 'item_visibility', 'item_type', 'item_mrp',
                               'outlet_establishment_year', 'outlet_size', 'outlet_location_type', 'outlet_type']

            inputs = {}
            for field in required_fields:
                value = request.form.get(field)
                if not value:
                    return render_template('error.html', error_message=f"Missing value for {field}")
                if field in ['item_weight', 'item_visibility', 'item_mrp', 'outlet_establishment_year']:
                    inputs[field] = float(value)
                else:
                    inputs[field] = value

            X = np.array([[inputs[field] for field in required_fields]])

            # Standardize input features
            X_std = sc.transform(X)

            # Make prediction
            Y_pred = model.predict(X_std)

            # Generate graph for predicted value
            predicted_value = float(Y_pred)
            df_pred = pd.DataFrame({
                'Feature': ['Predicted Sales'],
                'Value': [predicted_value]
            })

            fig_pred = px.bar(df_pred, x='Feature', y='Value', title='Predicted Sales Value')
            graph_pred_html = pio.to_html(fig_pred, full_html=False)

            # Generate graph for input features
            df_input = pd.DataFrame({
                'Feature': required_fields,
                'Value': [inputs[field] for field in required_fields]
            })

            fig_input = px.bar(df_input, x='Feature', y='Value', title='Input Features and Their Values')
            graph_input_html = pio.to_html(fig_input, full_html=False)

            return render_template('result.html', prediction=predicted_value, graph_input_html=graph_input_html, graph_pred_html=graph_pred_html)

        except ValueError as ve:
            return render_template('error.html', error_message=f"Value error: {ve}")

        except Exception as e:
            return render_template('error.html', error_message=str(e))
    else:
        return render_template('predict.html')

@app.route('/data', methods=['GET'])
def get_data():
    data = [
        {"date": "2023-01-01", "quantity": 100},
        {"date": "2023-02-01", "quantity": 150},
        {"date": "2023-03-01", "quantity": 200},
    ]
    return render_template('data.html', data=data)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
