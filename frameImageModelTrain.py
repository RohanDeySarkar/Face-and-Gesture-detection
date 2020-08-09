import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

h, w = 224, 224

path1 = 'data/detectPerson/rohan_data/'
path2 = 'data/detectPerson/intruder_data/'

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

base_model = Xception(
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

print(model.summary())

batch_size = 16
epochs = 5

datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

earlystop = EarlyStopping(monitor='val_loss')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
											patience=2,
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), validation_data = (X_test, y_test),
                    steps_per_epoch = len(X_train) / batch_size, epochs = epochs, callbacks = callbacks)

model.save("model_me.h5")

print("Training Done!")
