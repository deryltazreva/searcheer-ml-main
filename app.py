from flask import Flask, request, jsonify
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the model (make sure this points to the correct location)
model = tf.keras.models.load_model('model/capstone.h5')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get input data from request
        input_data = data['input']
        
       
        input_array = tf.convert_to_tensor(input_data)

        # Make prediction
        prediction = model.predict(input_array)
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
