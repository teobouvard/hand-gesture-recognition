import os
from glob import glob

from joblib import dump
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from image_preprocessing import flatten_image

if __name__ == "__main__":
    images = []
    targets = []

    for number in tqdm(os.listdir(os.path.join('data', 'train'))):
        files = glob(os.path.join('data', 'train', number, '*.jpg'))
        images.extend(map(flatten_image, files))
        targets.extend([number] * len(files))
    
    x_train, x_test, y_train, y_test = train_test_split(images, targets, test_size = 0.2, random_state = 42)

    model = LinearSVC(dual=False)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    model.fit(images, targets)
    dump(model, 'trained_model.joblib')
