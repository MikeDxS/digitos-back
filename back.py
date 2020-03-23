from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from resizer import resize_image, to_gray
from PIL import Image
import numpy as np
import wget
import os

modelo = tf.keras.models.load_model('./modelo.h5')
datos = np.zeros((1,28,28))
def procesar(url):
    wget.download(url, out='./img')
    img = Image.open('./img')
    img = img.convert('RGB')
    img.save('./img.jpg')
    resize_image('./img.jpg', './num.jpg', 28, 28, scale=False)
    to_gray('./num.jpg', './num.jpg')
    img = Image.open('./num.jpg')
    os.remove('./img.jpg')
    os.remove('./img')
    imgA = np.asarray(img, dtype=np.float32)
    dato = np.array([imgA])
    dato = abs(dato-255.0)/255.0
    return dato
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def saludar():
    saludo = 'Hola mundo'
    return jsonify({'saludo': saludo})

@app.route('/prediccion', methods=['POST'])
def predecir():
    url = request.get_json()['url']
    datos = procesar(url)
    prediccion = modelo.predict_classes(datos)
    print(prediccion)
    return jsonify({'url': url, 'prediccion': str(prediccion[0])})

if __name__ == '__main__':
    app.run(host='127.0.0.1')