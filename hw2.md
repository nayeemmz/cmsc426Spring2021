---
layout: page
mathjax: true
title: Image Classification Using Bag of Features and Support Vector Machines
permalink: /2020/hw/hw2/
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
11:59 PM, Monday, April 13, 2020

<a name='intro'></a>
## Introduction
In this homework you will implement Scale Invariant Feature Transform (SIFT). We will practice it on a set of imasges from Caltech-101 dataset that you would use in a later project as well. You may download Caltech-101 data
set from the following [link](http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download). All the images of this dataset are stored in folders, named for each category. However, we will be using just three of those categories: airplanes, dolphin and Leopards. If you would like to try for other categories too, please feel free to do so.  in Figure 1.

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

You will be implementing Scale-Invariant Feature Transform (SIFT) to obtain feature descriptors and not use a library for it. For the rest of the project you may use Python libaries. The descriptor for each image will be a matrix of size, $$keypoints \times 128$$. If there are different number of keypoints for different images, you may use only the strongest keypoints determined by the image having the smallest number of keypoints. Once the descriptors for each keypoint are obtained you may stack them for the entire training set. Use this matrix of feature descriptors as a training input to k-means clustering algorithm. The centroids of the clusters form a visual dictionary vocabulary. Use this visual vocabulary to make a frequency histogram for each image, based on the frequency of vocabularies in them. In other words you are trying to figure out the number of occurrences of each visual vocabulary word in each image. These histograms are the bag of visual words. The length of the histogram is the same as the number of clusters.

Go over the slides to understand SIFT, K-Means algorithm and bag of features. While you may use Python libraries to run K-means algorithm and to train the Support vector classifier, you would have to write your own code for SIFT. For a detailed description of SIFT, read the following [paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf). For the bag of visual words technique, follow the graphic above and read the following [paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/csurka-eccv-04.pdf).

[Here](/cmsc426fall2019/assets/proj3/regionalextrema.ipynb) is some starter code for extrema detection step.

## SVM Classifier Training

Train SVM on the resulting histograms (each histogram is a feature vector, with a label) obtained as a bag of visual words in the previous step. For a thorough understanding of SVM, refer to the heavily cited [paper](https://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf), by Christopher Burges.

You would need to train the classifiers as one vs. all. Wherein only the category that you are training for is considered to be a positive example and the other two categories are treated as negative examples. You may use svm from sklearn in Python.

## Test your model

Extract the bag of visual words for the test image and then pass it as an input to the SVM models you created during
training to predict its label. That means it would be tested using all the SVM classifiers and assigned the label that gives the highest score.

<a name='part2'></a>
## Part 2: - What to submit (50 points)

1. Show a 3 x 3 confusion matrix with categories as its rows and columns. It is used to determine the
accuracy of your classifier. In this matrix the rows are the actual category label and the columns are the predicted
label. Each cell in this matrix will contain the prediction count. Ideally, we would like all the off-diagonal
numbers in this matrix to be 0’s, however, that is not always possible. For example in the matrix below with
100 images of each of the three categories, airplanes, dolphin, Leopards,
<div class="fig fighighlight">
  <img src="/cmsc426fall2019/assets/proj3/confusion.png" width="50%">
  <div class="figcaption">
  </div>
  <div style="clear:both;"></div>
</div>
the confusion matrix can be read as, airplane was correctly classified as an airplane, 93 times, and wrongly classified as
dolphin and leopard, two times and five times, respectively. Similarly, dolphin was correctly classified 98 out of 100 times
and leopard was also correctly classified 98% of the time.

2. A plot showing the histogram of the visual vocabulary during the training phase. You can pick any image you
like.

3. Some of the image patches corresponding to the words in the visual vocabulary (cluster centroids).


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
You are encouraged to work in groups for this project. You may discuss the ideas with your peers from other groups. If you reference anyone else's code in writing your project, you must properly cite it in your code (in comments) and your writeup.  For the full honor code refer to the CMSC426 Fall 2019 website
