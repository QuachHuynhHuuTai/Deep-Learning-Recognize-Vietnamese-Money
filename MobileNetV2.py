from os import listdir
import pandas as pd
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import itertools
import random
from keras.preprocessing.image import ImageDataGenerator

raw_folder = "data/"
def save_data(raw_folder=raw_folder):

    print("Image processing...")

    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):
        if folder!='.DS_Store':
            print("Folder = ",folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    print("File = ", file)
                    pixels.append( cv2.resize(cv2.imread(raw_folder  + folder +"/" + file),dsize=(128,128)))
                    labels.append( folder)

    pixels = np.array(pixels)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('money.data', 'wb')
    # dump information to that file
    pickle.dump((pixels,labels), file)
    # close the file
    file.close()

    return

def load_data():
    file = open('money.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)


    return pixels, labels

# save_data()
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

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

mbnv2_model = get_model()

filepath="MobileNetV2-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    rescale=1./255,
	width_shift_range=0.1,
    height_shift_range=0.1,
	horizontal_flip=True,
    brightness_range=[0.2,1.5], fill_mode="nearest")


aug_val = ImageDataGenerator(rescale=1./255)

mbnv2hist=mbnv2_model.fit_generator(aug.flow(X_train, y_train, batch_size=64),
                               epochs=3,# steps_per_epoch=len(X_train)//64,
                               validation_data=aug.flow(X_val,y_val,
                               batch_size=64),
                               callbacks=callbacks_list)

mbnv2_model.save("MobileNetV2model3e.h5")



loss_train = mbnv2hist.history['loss']
loss_val = mbnv2hist.history['val_loss']
epochs = range(1, 201)
plt.plot(epochs, loss_train, 'b', label='Training loss')
plt.plot(epochs, loss_val, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train = mbnv2hist.history['accuracy']
acc_val = mbnv2hist.history['val_accuracy']
epochs = range(1, 201)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
