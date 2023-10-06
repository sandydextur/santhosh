import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.activations import softmax


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('leo2.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    image = request.files[accept]
    img = tf.keras.utils.load_img(image ,target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ["actinic", "melonama", "acne"]

  

    output = class_names[np.argmax(score)]
    return render_template('sandy.html', prediction_text='{}'.format(output))
    

if __name__ == "__main__":
    application.run(debug=True)