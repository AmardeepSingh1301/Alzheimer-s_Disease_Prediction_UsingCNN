from flask import Flask, request,render_template

import os;

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':

        image_data = request.files["imgInp"]

        if(not image_data):

            return render_template('index.html',output="Please Choose Image")
        
        directory = os.path.join('./Upload/',image_data.filename);

        #model
        import cv2
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras import models
        import numpy as np
        model = tf.keras.models.load_model('alzheimers.h5',compile=False)

        img=cv2.imread(directory)
        img=cv2.resize(img,(150,150))
        img_array=np.array(img)
        # img_array.shape

        img_array=img_array.reshape(1,150,150,3)
        # img_array.shape

        a=model.predict(img_array)
        indices=a.argmax()
        if indices==0:
            o="Mild Demented"
        elif indices==1:
            o="Moderate Demented"
        elif indices==2:
            o="Non Demented"
        else:
            o="Very Mild Demented"

        return render_template('index.html',output=o)
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run()