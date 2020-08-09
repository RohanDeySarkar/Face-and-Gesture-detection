import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


h, w = 224, 224

path1 = 'data/detectGesture/v_data/'
path2 = 'data/detectGesture/blank_data/'

train_images, train_labels = [], []

for i in range(len(os.listdir(path1))):
	train_image = cv2.imread(path1 + str(i) + '.png')
	train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
	train_image = cv2.resize(train_image, (h, w))
	train_images.append(train_image)
	train_labels.append(0)

for i in range(len(os.listdir(path2))):
	train_image = cv2.imread(path2 + str(i) + '.png')
	train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
	train_image = cv2.resize(train_image, (h, w))
	train_images.append(train_image)
	train_labels.append(1)

train_images = np.array(train_images)

X_train, X_test, y_train, y_test = train_test_split(train_images, to_categorical(train_labels), test_size=0.2, random_state=42)

base_model = ResNet50(
    weights='imagenet',
    include_top=False, 
    input_shape=(h, w, 3), 
    pooling='avg'
)

base_model.trainable = False

model = Sequential([
  base_model,
  Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16
epochs = 3

datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)


model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), validation_data = (X_test, y_test),
                    steps_per_epoch = len(X_train) / batch_size, epochs = epochs)


test_images = train_images[90:110]
test_labels = train_labels[90:110]
labels = model.predict(test_images)
labels = [np.argmax(i) for i in labels]
print(labels)
print("------------------------------------------------------------")
print(test_labels)

# model.save("model.h5")

print("Training Done!")