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
    features = [
        data['lotArea'],
        data['yearBuilt'],
        data['overallQual'],
        data['totalBsmtSF'],
        data['firstFlrSF'],
        data['grLivArea'],
        data['fullBath'],
        data['bedroomAbvGr'],
        data['kitchenQual'],
        data['garageCars'],
        data['garageArea']
    ]
    
    # Convert categorical kitchen quality to numerical
    kitchen_quality_mapping = {'TA': 1, 'Gd': 2, 'Ex': 3}
    features[8] = kitchen_quality_mapping[features[8]]
    
    final_features = np.array([features])
    prediction = model.predict(final_features)
    
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
