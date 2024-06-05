from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    app.logger.info(f"Received data: {data}")

    try:
        # Convert input data to correct types
        data['yearBuilt'] = int(data['yearBuilt'])
        data['lotArea'] = float(data['lotArea'])
        data['garageCars'] = int(data['garageCars'])
        data['totRmsAbvGrd'] = int(data['totRmsAbvGrd'])
        data['fullBath'] = int(data['fullBath'])
        data['halfBath'] = int(data['halfBath'])
        
        # Convert keys to match the expected format
        data = {key.title().replace('_', ''): value for key, value in data.items()}
        
        # Convert input data to DataFrame
        features = pd.DataFrame([data], columns=[
            'Neighborhood', 'YearBuilt', 'LotArea', 'BldgType', 'CentralAir', 
            'GarageCars', 'TotRmsAbvGrd', 'FullBath', 'HalfBath'
        ])
        
        app.logger.info(f"Features before processing: {features}")

        # Preprocess features
        processed_features = model.named_steps['preprocessor'].transform(features)
        app.logger.info(f"Processed features: {processed_features}")

        # Make prediction
        prediction = model.named_steps['regressor'].predict(processed_features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
