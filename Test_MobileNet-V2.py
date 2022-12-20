from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import load_model
import sys

cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name = ['00000','10000','100000','20000','200000','5000','50000','500000']

def get_model():
    model_mbnv2_conv = MobileNetV2(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_mbnv2_conv.layers[:-4]:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_mbnv2_conv = model_mbnv2_conv(input)

    # Them cac layer FC va Dropout
    x = output_mbnv2_conv

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)  # dense layer 3
    x = Dropout(0.2)(x)
    x = Dense(8, activation='softmax')(x)  # final layer with softmax activation for N classes


    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights model da train
my_model = get_model()
my_model.load_weights("MobileNetV2-140-0.99.hdf5")

while(True):
    # Capture frame-by-frame
    #

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):


        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()