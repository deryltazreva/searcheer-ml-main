from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Muat model
model = joblib.load('model/tokenizer.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["input"]])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)

