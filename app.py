from flask import Flask, request, jsonify
from bentos_model.src.detection_model import Detection_model
import base64

app = Flask(__name__)
model = Detection_model()

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'URL not provided in JSON data'}), 400

    image_urls= data['url']
    data = model.predict(image_urls)
    return data

if __name__ == 'main':
    app.run(debug=True)