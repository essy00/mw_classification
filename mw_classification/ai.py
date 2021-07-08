import os
import random

from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Flatten,
    Dropout,
    MaxPooling2D
)

img_size = 64

categories = ['men', 'women']
base_dir = "./data"

data = []

# takes the data, makes it gray and resizes it.
for category in categories:
    folder_path = os.path.join(base_dir, category)
    value = categories.index(category)
    print(f'{category}: {value}')
    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (img_size, img_size))
            data.append([image, value])

# man 0
# woman 1

X = []
Y = []

# in order for the machine to not memorize the data
random.shuffle(data)
for x, y in data:
    X.append(x)
    Y.append(y)

# for a better performance
del data

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
Y = np.array(Y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.33,
    random_state=42
)

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X[0].shape, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=200, shuffle=True)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

model.save('model.h5')
