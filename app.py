from flask import Flask
import base64

app = Flask(__name__)

@app.route('/predict')
def predict():
    dummy_img = open('dummy.jpg', 'rb')

    data = {
        'predicted class': "Dreissena polymorpha",
        'image': base64.b64encode(dummy_img.read()).decode()
    }

    return data

# if __name__ == 'main':
app.run(debug=True)