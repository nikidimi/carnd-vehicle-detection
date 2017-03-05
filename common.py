import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

def bin_spatial(img, size=(16, 16)):
    features = cv2.resize(img, size).ravel() 
    return features

def color_hist(img, nbins=16, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    return [channel1_hist[0], channel2_hist[0], channel3_hist[0]]

def get_hog(image_channel, feature_vector=True):
    orient = 9 
    pix_per_cell = 8
    cell_per_block = 2
    return hog(image_channel, orient, (pix_per_cell, pix_per_cell),
               (cell_per_block, cell_per_block), feature_vector=feature_vector)
               
def get_image_features_arr(image, hsv):
    features = [];

    
    features.append(get_hog(hsv[:, :, 0]))
    features.append(get_hog(hsv[:, :, 1]))
    features.append(get_hog(hsv[:, :, 2]))
    features.append(bin_spatial(hsv))
    features.extend(color_hist(hsv))

    return np.concatenate(features)


def get_image_features(image_path):
    image = mpimg.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    
    return get_image_features_arr(image, hsv)
