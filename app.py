from flask import Flask, request, jsonify
from src.ML_model import ML_model
import base64

app = Flask(__name__)
model = ML_model()

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'URL not provided in JSON data'}), 400

    image_urls= data['url']
    data = model.handle_images(image_urls)
    return jsonify(data)

if __name__ == 'main':
    app.run(debug=True)