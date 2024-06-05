from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

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
        features = [
            data['neighborhood'],
            data['lotArea'],
            data['yearBuilt'],
            data['bldgType'],
            data['centralAir'],
            data['garageCars'],
            data['totRmsAbvGrd'],
            data['fullBath'],
            data['halfBath']
        ]

        # Convert features into the appropriate format
        final_features = np.array([features])
        prediction = model.predict(final_features)
        app.logger.info(f"Prediction: {prediction[0]}")
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
