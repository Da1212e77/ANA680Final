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
    try:
        # Convert input data to DataFrame and ensure correct types
        features = pd.DataFrame([data], columns=[
            'neighborhood', 'lotArea', 'yearBuilt', 'bldgType', 'centralAir', 
            'garageCars', 'totRmsAbvGrd', 'fullBath', 'halfBath'
        ])
        
        # Convert numerical columns to correct type
        features['lotArea'] = features['lotArea'].astype(float)
        features['yearBuilt'] = features['yearBuilt'].astype(int)
        features['garageCars'] = features['garageCars'].astype(int)
        features['totRmsAbvGrd'] = features['totRmsAbvGrd'].astype(int)
        features['fullBath'] = features['fullBath'].astype(int)
        features['halfBath'] = features['halfBath'].astype(int)
        
        # Preprocess features
        processed_features = preprocessor.transform(features)

        # Make prediction
        prediction = model.predict(processed_features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
