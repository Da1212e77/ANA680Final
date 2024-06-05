from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        
        # Preprocess features
        features = preprocess_features(features)

        # Make prediction
        prediction = model.predict(features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

def preprocess_features(df):
    # Define categorical and numerical features
    categorical_features = ['neighborhood', 'bldgType', 'centralAir']
    numerical_features = ['lotArea', 'yearBuilt', 'garageCars', 'totRmsAbvGrd', 'fullBath', 'halfBath']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor.fit_transform(df)

if __name__ == '__main__':
    app.run(debug=True)
