---
layout: page
mathjax: true
title: Image Classification Using Bag of Features and Support Vector Machines
permalink: /2019/proj/p3/
---



Table of Contents:
- [Due Date](#due)
- [Introduction](#intro)
- [Part 1: Implementation](#part1)
- [Part 2: What to submit](#part2)
- [Submission Guidelines](#sub)
- [Collaboration Policy](#coll)

<a name='due'></a>
## Due Date 
11:59 PM, Sunday, November 24, 2019

<a name='intro'></a>
## Introduction
In this homework you will implement an image classifier.You will be building Support Vector Machine (SVM)
classifier to classify images of Caltech-101 dataset.
Supervised classification is a computer vision task of categorizing unlabeled images to different categories or
classes. This follows the training using labeled images of the same categories. You may download Caltech-101 data
set from the following [link](http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download). All the images of this dataset are stored in folders, named for each category. However, we will be using just three of those categories: airplanes, dolphin and Leopards. Since there are fewer dolphins than the other categories, we will use same number of images for the other categories as well. You would
use these labeled images as training data set to train SVM classifier after obtaining a bag (histogram) of visual words for each image. The classification would be one-vs-all, where
you would specifically consider one image at a time to classify and consider it as a positive example and all other
category images as negative examples. Once the classifier is trained you would test an unlabeled image and classify
it as one of the three categories. This task can be visualized in Figure 1

<div class="fig fighighlight">
  <img src="/cmsc426fall2019/assets/proj3/proj3.png" width="100%">
  <div class="figcaption">
  </div>
  <div style="clear:both;"></div>
</div>


<a name='part1'></a>
## Part 1: Implementation (50 pts)


There are three major steps in this approach.

## Creating bag of visual words

You will be implementing Scale-Invariant Feature Transform (SIFT) to obtain feature descriptors. You may use Python libraries to carry out other parts of the project but not SIFT. The descriptor for each image will be a matrix of size, $`keypoints \times 128'$. If there are different number of keypoints for different images, you may use only the strongest keypoints determined by the image having the smallest number of keypoints. You may Stack the matrices of these descriptors for the entire training set. Cluster these feature descriptors using k-means clustering algorithm. Since you may not obtain the same number of kepoints for each image of the training dataset, use the strongest keypoints equivalebt to the image with the least number of keypoints. Use the centroids of the clusters to form a visual dictionary vocabulary. Use this visual vocabulary to make a frequency histogram for each image, based on the frequency of vocabularies in them. In other words you are trying to figure out the number of occurrences of each visual vocabulary word in each image. These histograms are the bag-of-visual-words. The length of the histogram is the same as the number of clusters.

Go over the slides to understand SIFT, K-Means algorithm and bag of features. While you may use Python libraries to run k-means algorithm and to train the Support vector classifier, you would have to write your own code for SIFT. For a detailed description of SIFT, read the the paper, “Distinctive Image Features from Scale-Invariant Keypoints" by David G. Lowe. 

## SVM Classifier Training

Train SVM on the resulting histograms (each histogram is a feature vector, with a label) obtained as a bag of visual words in the previous step. For a thorough understanding of SVM, refer to the heavily cited paper, “A Tutorial on
Support Vector Machines for Pattern Recognition", by Christopher Burges. 

You would need to train the classifier as a one vs. all. Wherein only the digit that you are training for is considered to bt a positive example and every other digit is treated as a negative example. You may use svm from sklearn in Python. 

## Test your model

Apply the trained classifier to the test image. Here you would test it the following two ways:

- Extract the bag of visual words for the test image and then pass it as an input to the model you created during
training, and,
- Pass the test image directly to the trained SVM model without any feature extraction.



<a name='part2'></a>
## Part 2: - What to submit (50 points)

1. Show a 10 × 10 confusion matrix with digits from 0-9 as its rows and columns. It is used to determine the
accuracy of your classifier. In this matrix the rows are the actual digit label and the columns are the predicted
label. Each cell in this matrix will contain the prediction count. Ideally, we would like all the off-diagonal
numbers in this matrix to be 0’s, however, that is not always possible. For example in the matrix below with
100 images of each of the three digits, 0, 1, 2,
<div class="fig fighighlight">
  <img src="/cmsc426fall2019/assets/hwk3/table.png" width="50%">
  <div class="figcaption">
  </div>
  <div style="clear:both;"></div>
</div>
the confusion matrix can be read as, digit 0 was correctly classified as a 0, 93 times, and wrongly classified as
1 and 2, two times and five times, respectively. Similarly, digit 1 was correctly classified 98 out of 100 times
and digit 2 was also correctly classified 98% of the time.

2. A plot showing the histogram of the visual vocabulary during the training phase. You can pick any digit you
like.

3. Images of visual vocabulary for some of the clusters.


<a name='sub'></a>
## Submission Guidelines

File tree and naming
Your submission on Canvas must be a zip file, following the naming convention YourDirectoryID_proj3.zip. For
example, xyz123_proj3.zip. The file must have the following directory structure, based on the starter files
- mysvm.ipynb
- report.pdf


### Report

Please include the plot and confusion matrix as mentioned in part 2. Also include your observations about the
prediction of test images.
As usual, your report must be full English sentences,not commented code

<a name='coll'></a>
## Collaboration Policy
You are encouraged to work in groups for this project. You may discuss the ideas with your peersfrom other groups. If you reference anyone else's code in writing your project, you must properly cite it in your code (in comments) and your writeup.  For the full honor code refer to the CMSC426 Fall 2019 website
