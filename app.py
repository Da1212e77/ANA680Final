import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('house_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure all expected fields are present
        expected_fields = ['neighborhood', 'yearBuilt', 'lotArea', 'bldgType', 'centralAir', 'garageCars', 'totRmsAbvGrd', 'fullBath', 'halfBath']
        for field in expected_fields:
            if field not in data:
                raise ValueError(f'Missing field: {field}')

        # Map the incoming data to the expected format for the model
        input_data = {
            'Neighborhood': data['neighborhood'],
            'YearBuilt': int(data['yearBuilt']),
            'LotArea': int(data['lotArea']),
            'BldgType': data['bldgType'],
            'CentralAir': data['centralAir'],
            'GarageCars': int(data['garageCars']),
            'TotRmsAbvGrd': int(data['totRmsAbvGrd']),
            'FullBath': int(data['fullBath']),
            'HalfBath': int(data['halfBath'])
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_df)
        
        # Make a prediction
        prediction = model.predict(processed_data)
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

