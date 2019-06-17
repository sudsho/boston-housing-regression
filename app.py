"""Tiny Flask app that serves housing price predictions."""
from flask import Flask, request, jsonify

from src.predict import predict_one
from src.data import FEATURE_NAMES


app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({'error': 'expected JSON body'}), 400

    missing = [c for c in FEATURE_NAMES if c not in payload]
    if missing:
        return jsonify({'error': 'missing fields', 'missing': missing}), 400

    record = {c: float(payload[c]) for c in FEATURE_NAMES}
    pred = predict_one(record)
    return jsonify({'medv': pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
