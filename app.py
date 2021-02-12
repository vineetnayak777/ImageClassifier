from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os

import tensorflow as tf
import numpy as np
from tensorflow import keras

UPLOAD_FOLDER = "static/"

app =  Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

class_names =['n02085620-Chihuahua', 'n02086240-Shih-Tzu', 'n02088364-beagle']
img_width = 100
img_height = 100

model = tf.keras.models.load_model('dog-breed.h5')

def get_prediction(file_name):
    
    img = keras.preprocessing.image.load_img(
        file_name,
        target_size = (img_width, img_height)
    )   

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    predicted_name = class_names[np.argmax(score)]
    per = np.max(score) * 100

    return predicted_name, per

@app.route("/", methods=['GET','POST'])
def upload_and_predict():

    if request.method == "POST":
        file_obj = request.files['file']

        if file_obj:
            f_name = secure_filename(file_obj.filename)
            f_name = os.path.join(UPLOAD_FOLDER, f_name)
            file_obj.save(f_name)

            name, per = get_prediction(f_name)

            return render_template("index.html", img = f_name, img_class_name = name, accuracy = per)

    return render_template("index.html")

if __name__ == "__main__":
    app.run()