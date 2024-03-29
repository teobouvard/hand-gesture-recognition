import os
from glob import glob

import cv2
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

from image_preprocessing import extract_hand


if __name__ == "__main__":
    features = []
    targets = []

    # read images and populate features/targets
    for i in range(6):
        files = glob(os.path.join('data', 'train', str(i), '*.jpg'))
        for f in files:
            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            features.append(image)
            targets.append(i)

    features = np.array(features)
    targets = np.array(targets)

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)
    
    # build and compile DNN
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(50, 50)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    # save initial weights for full training
    initial_weights = model.get_weights()

    # fit on training set and evaluate model
    model.fit(x_train, y_train, epochs=3)
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'Accuracy on {len(x_test)} test samples: {test_acc}')

    # reset the model weights before training on the whole dataset
    model.set_weights(initial_weights)
    model.fit(features, targets, epochs=3)
    model.save('trained_classifier.h5')
