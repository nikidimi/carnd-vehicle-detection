[//]: # (Image References)
[image1]: ./out5.jpg
[image2]: ./out6.jpg

# Rubric Points


### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting various features is located in common.py

Extraction of HOG features is located in the get_hog function

I'm using the hog function from skimage.features.

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters and the color space were chosen after experimenting with different values. Using this configuration resulted in the highest accuracy

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

 The training of the classifier is done in train.py.

* First, I read all the images and extract the features from them. This done in multiple processes in order to speed-up the training.
* After that the features are scaled using StandardScaler from sklearn.
* The final step is to run the classifier. I'm using the SVC linear classifier
* In the end, the trained classifier and the scaler are saved to files

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window is implemented in the function calc_rect in detect.py

First, I calculate the HOG features for the whole image. After that, I iterate through the result with a step of 2. In order to get the corresponding subimage to the hog features, I multiple the x and y coordinates by the pixels per cell values - 8.

I've tried different scales and it end decided to keep 1.0, 1.5, 2.0 and 3.0. They do a reasonable job of finding cars in different positions. I always tried different steps and 2 seems to be an optimal between not having too many duplicates and capturing all the cars.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I've tuned the parameters of the feature extractors. I've also added decision function threshold.

Here are some pictures before applying the heatmap:
![alt text][image1]
![alt text][image2]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_aws.mp4)

* The upper-left corner of the video is the final result.
* The upper-right corner show the results before applying the heatmap.
* In the bottom, the heatmap itself is displayed

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Overlapping bounding boxes are combined by using a heatmap. I retain the last 10 heatmaps and combine them in order to smooth the results.
This is implemented in the function parse_image in detect.py

False positives are filtered by thresholding the decision function and by thresholding the heatmap


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The training of the classifier is rather slow and tuning all the parameters takes a lot of trail and error.

It is possible to make it more robust by adding some kind of tracking for the cars - the can only move a few pixels per frame, their scale should be determined by their position in the frame and they can't just pop up in the middle of the road. By checking for all this conditions, we can remove false positives.

We can also make a more rigorous search (by reducing the classifier threshold, searching with more scales and etc) if a car suddenly disappears.
