from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = 'house_price_model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    app.logger.info(f"Received data: {data}")
    try:
        # Create a DataFrame with the appropriate column names
        features = pd.DataFrame([data], columns=[
            'neighborhood', 'lotArea', 'yearBuilt', 'bldgType', 'centralAir', 
            'garageCars', 'totRmsAbvGrd', 'fullBath', 'halfBath'
        ])
        
        # Make prediction
        prediction = model.predict(features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
