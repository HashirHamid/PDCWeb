import tensorflow as tf
import os

import pymysql
import werkzeug.utils
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import cv2
from flask import Flask,render_template, jsonify, request
import json



'''

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True) #aik image ko different angle dene k le kiu k aik hi image br br dekh k model overfit ho je ga
train_images = "C:/Users/Usama Ejaz Wais/Desktop/chest_xray/train"
train_generator = train_datagen.flow_from_directory(train_images,target_size = (300,300),batch_size = 128,class_mode = 'binary')
print(train_generator.class_indices)

test_datagen = ImageDataGenerator(rescale = 1./255)

#Validation data generator and loading validation data
validation_generator = test_datagen.flow_from_directory('C:/Users/Usama Ejaz Wais/Desktop/chest_xray/val',
    target_size= (300,300),
    batch_size = 128,
    class_mode = 'binary')

#All layers are given in form of list

model= tf.keras.models.Sequential([
                                   tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape= (300, 300, 3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),

                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(256, activation= 'relu'),
                                   tf.keras.layers.Dense(512, activation= 'relu'),
                                   tf.keras.layers.Dense(1, activation= 'sigmoid')
])
model.summary()
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

Training_model = model.fit_generator(train_generator,  epochs = 20, validation_data = validation_generator)

model.save("trained.h5")

'''

model = load_model("trained.h5")


def check(img1):
    eval_datagen = ImageDataGenerator(rescale=1 / 255)

    test_generator = eval_datagen.flow_from_directory(
        'D:/University/Semester 5/PDCWeb/chest_xray/test',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )
    img= cv2.imread('D:/University/Semester 5/PDCWeb/New folder/uploadedImages/'+img1)
    tempimg = img
    img = cv2.resize(img,(300,300))
    img = img/255.0
    img = img.reshape(1,300,300,3)
    print( model.predict(img))

    result = ''
    prediction = model.predict(img) >= 0.5
    if prediction>=0.5:
        prediction = "Pneumonia"
        result = '1'
    else:
        prediction = "Normal"
        result = '0'

    print(prediction)

    return result


app = Flask(__name__)
response = ''

@app.route("/hello",methods = ['GET', 'POST'])
def index():

    if(request.method == 'POST'):
        imageFile = request.files['file']
        filename = werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save('D:/University/Semester 5/PDCWeb/New folder/uploadedImages/'+filename)
        return check(filename)
    return 'hello'


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port='5000')





