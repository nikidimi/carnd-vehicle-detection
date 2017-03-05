import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from multiprocessing import Pool 
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from common import get_image_features

def prepare_data(car_images, non_car_images):
    pool = Pool() 
    X_car = pool.map(get_image_features, car_images)
    Y_car = [1] * len(X_car)
    X_non_car = pool.map(get_image_features, non_car_images)
    Y_non_car = [0] * len(X_non_car)

    X = X_car + X_non_car
    Y = Y_car + Y_non_car
    
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    
    with open('scaler.pickle', 'wb') as handle:
        pickle.dump(X_scaler, handle)  

    return scaled_X, Y

def train(X, Y):
    X, Y = shuffle(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    
    print("Training with {} examples, testing with {}". format(len(X_train), len(X_test)))
    print("Feature vecture len: {}".format(len(X_train[0])))

    clf = LinearSVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    print("Accuracy:{}".format(accuracy_score(y_test, y_pred)))
    
    with open('clf.pickle', 'wb') as handle:
        pickle.dump(clf, handle)
    
if __name__ == "__main__":
    car_images = glob.glob('train/vehicles/*/*.png')
    non_car_images = glob.glob('train/non-vehicles/*/*.png')

    car_images = shuffle(car_images)
    non_car_images = shuffle(non_car_images)    

    X, Y = prepare_data(car_images, non_car_images)
    train(X, Y)
