from flask import Flask, request, jsonify
from src.model import Detection_model
import base64

app = Flask(__name__)
model = Detection_model()

@app.route('/predict')
def predict():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'URL not provided in JSON data'}), 400

    image_url = data['url']
    data = model.predict(image_url)
    return data

if __name__ == 'main':
    app.run(debug=True)