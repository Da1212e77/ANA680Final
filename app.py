from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('house_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    app.logger.info(f"Received data: {data}")

    # Convert keys to PascalCase
    data = {key.title().replace('_', ''): value for key, value in data.items()}

    try:
        # Convert input data to DataFrame and ensure correct types
        features = pd.DataFrame([data], columns=[
            'Neighborhood', 'YearBuilt', 'LotArea', 'BldgType', 'CentralAir', 
            'GarageCars', 'TotRmsAbvGrd', 'FullBath', 'HalfBath'
        ])
    
        print("Data:", data)
        print("Features before processing:", features)
    
        # Convert numerical columns to correct type
        features['LotArea'] = features['LotArea'].astype(float)
        features['YearBuilt'] = features['YearBuilt'].astype(int)
        features['GarageCars'] = features['GarageCars'].astype(int)
        features['TotRmsAbvGrd'] = features['TotRmsAbvGrd'].astype(int)
        features['FullBath'] = features['FullBath'].astype(int)
        features['HalfBath'] = features['HalfBath'].astype(int)
    
        # Preprocess features
        processed_features = preprocessor.transform(features)

        print("Processed features:", processed_features)

        # Make prediction
        prediction = model.predict(processed_features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
