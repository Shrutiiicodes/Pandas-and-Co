#  flask prediction
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    pred = model.predict([data])
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)


# for testing
# curl -X POST http://127.0.0.1:5000/predict \
# -H "Content-Type: application/json" \
# -d '{"features":[1,2,3,4]}'
