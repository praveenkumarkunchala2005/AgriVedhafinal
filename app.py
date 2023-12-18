import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import six.moves.urllib as urllib
import sys
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# Load the trained model
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the pre-trained model
model_filename = "PlantDNet.h5"
model_path = "PlantDNet.h5"
model = tf.keras.models.load_model(model_path)

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict2', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [data['N'], data['P'], data['K']]
        prediction = loaded_model.predict([input_data])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/index1', methods=['GET'])
def index1():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = preds[0]
        ind = np.argmax(a)
        result = disease_class[ind]
        return result
    return None

if __name__ == '__main__':
    # Use Gunicorn as the WSGI server
    # gunicorn -w 4 app:app
    app.run()
