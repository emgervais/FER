import sys, os
import pandas as pd
import numpy as np
import joblib as jb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy, MeanSquaredError, SparseCategoricalCrossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Flatten, Input, Activation
import cv2 as cv
import matplotlib.pyplot as plt

num_features = 64
num_labels = 8
batch_size = 64
epochs = 10
np.set_printoptions(threshold=sys.maxsize)

def normalize(data):
    return (data / 255.0).reshape(data.shape[0], 64, 64, 1)

def NormalizeData(data):
    return (data / 10)

fer2013 = pd.read_csv("fer2013.csv")
ferplus_labels = pd.read_csv("label.csv")
ferplus_labels.pop("NF")
ferplus_labels.pop("unknown")

X_train, train_y, X_test, test_y = [], [], [], []

for index, row in fer2013.iterrows():
    val = row['pixels'].split(" ")
    try:
        image_name = f"fer{index:07d}.png"
        label_row = ferplus_labels[ferplus_labels['Image name'] == image_name]
        
        if not label_row.empty:
            if label_row.iloc[0]['Usage'] == 'Training':
                X_train.append(cv.resize(np.array(val, 'float32').reshape(48, 48, 1), (64, 64)))
                train_y.append(np.array(label_row.iloc[0, 2:], 'float32'))
            elif label_row.iloc[0]['Usage'] == 'PublicTest':
                X_test.append(cv.resize(np.array(val, 'float32').reshape(48, 48, 1), (64, 64)))
                test_y.append(np.array(label_row.iloc[0, 2:], 'float32'))
    except:
        print(f"Error occurred at index: {index} for row: {row}")

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

X_train = normalize(X_train)
X_test = normalize(X_test)
train_y = NormalizeData(train_y)
test_y = NormalizeData(test_y)

model = Sequential()

# Input Layer for 64x64 images
model.add(Input(shape=(64, 64, 1)))

# Conv Layer 1
model.add(Conv2D(int(num_features / 2), kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Output size: 32x32

# Conv Layer 2
model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Output size: 16x16

# Conv Layer 3
model.add(Conv2D(num_features * 2, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Output size: 8x8

# Conv Layer 4
model.add(Conv2D(num_features * 4, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Output size: 4x4

# Flatten for Dense layers
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (assuming softmax for 7 categories)
model.add(Dense(num_labels, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# Train the model
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)

# Save the model
jb.dump(model, "model")

# Evaluate the model
train_score = model.evaluate(X_train, train_y, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100 * train_score[1])
test_score = model.evaluate(X_test, test_y, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100 * test_score[1])
