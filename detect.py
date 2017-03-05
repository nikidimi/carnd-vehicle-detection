import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from multiprocessing.dummy import Pool as ThreadPool 
from scipy.ndimage.measurements import label
from sklearn.svm import SVC, LinearSVC
import pickle
from moviepy.editor import VideoFileClip, ImageSequenceClip
from common import get_image_features, get_hog, bin_spatial, color_hist, get_image_features_arr

heatmaps = []

def calc_rect(image, rect, scale):
    global clf

    scaled = cv2.resize(image, (np.int(image.shape[1]/scale), np.int(image.shape[0]/scale)))
    scaled = (scaled/255.0).astype('float32')
    hsv = cv2.cvtColor(scaled, cv2.COLOR_RGB2HSV)
    
    h1 = get_hog(hsv[:, :, 0], feature_vector=False)
    h2 = get_hog(hsv[:, :, 1], feature_vector=False)
    h3 = get_hog(hsv[:, :, 2], feature_vector=False)
    
    
    for x in range(0, h1.shape[0] - 7, 2):
        for y in range(0, h1.shape[1] - 7, 2):
            subimage = hsv[(x * 8): (x * 8) + 64, (y * 8): (y * 8) + 64, :]
            
            features = []
            features.append(h1[x:x + 7, y:y + 7].ravel())
            features.append(h2[x:x + 7, y:y + 7].ravel())
            features.append(h3[x:x + 7, y:y + 7].ravel())
            features.append(bin_spatial(subimage))
            features.extend(color_hist(subimage))
            
            data = np.concatenate(features)
            scaled_data = X_scaler.transform(data.reshape(1, -1))
            
            result = clf.predict(scaled_data)

            if result[0] > 0 and clf.decision_function(scaled_data) > 0.6:
                rect.append(((np.int(y * 8 * scale), 
                              np.int(360 + x * 8 * scale)), 
                             (np.int((y+7) * 8 * scale), 
                              np.int(360 + (x+7) * 8 * scale))))

def draw_labeled_bboxes(image, labels):
    img = np.copy(image)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def parse_image(image):
    global heatmaps     
    output = np.empty((1440, 2560, 3), dtype='uint8')
    
    bboxes = []   
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    
    for scale in [1.0, 1.5, 2.0, 3.0]:
        calc_rect(image[360:720], bboxes, scale)
        
    for box in bboxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        
    heatmaps.append(heatmap)
    if len(heatmaps) > 10:
        heatmaps.pop(0)
    
    summed = np.sum(heatmaps, 0)
    threshold_heatmap = np.copy(summed)
    threshold_heatmap[threshold_heatmap <= 5] = 0
    labels = label(threshold_heatmap)
    
            
    output[0:720, 0:1280] = draw_labeled_bboxes(image, labels)
    
    output[720:1440, 0:1280, 0] = summed * 10
    output[720:1440, 0:1280, 1] = summed * 10
    output[720:1440, 0:1280, 2] = summed * 10
    
    output[720:1440, 1280:2560, 0] = heatmap * 10
    output[720:1440, 1280:2560, 1] = heatmap * 10
    output[720:1440, 1280:2560, 2] = heatmap * 10
    
    output[0:720, 1280:2560] = draw_boxes(image, bboxes)
    
    return output

if __name__ == "__main__":
    global clf, X_scaler
    
    with open('clf.pickle', 'rb') as handle:
        clf = pickle.load(handle)
    with open('scaler.pickle', 'rb') as handle:
        X_scaler = pickle.load(handle)
    
    clip = VideoFileClip("project_video.mp4")
    processed_clip = clip.fl_image(parse_image)
    processed_clip.write_videofile("out.mp4", audio=False)
