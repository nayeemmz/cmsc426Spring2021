---
layout: page
mathjax: true
title: Color Segmentation using GMM
permalink: /2019/hw/hw3/
---

Table of Contents:
- [Deadline](#due)
- [Introduction](#intro)
- [What you need to do](#problem)
  - [Problem Statement](#pro)
- [Submission Guidelines](#sub)
- [Collaboration Policy](#coll)

<a name='due'></a>
## Deadline
11:59PM, Saturday, Oct 26, 2019

<a name='intro'></a>
## Introduction

Have you ever played with these adorable Nao robots? Click on the image to watch a cool demo.  

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Gy_wbhQxd_0
" target="_blank"><img src="http://img.youtube.com/vi/Gy_wbhQxd_0/0.jpg"
alt=" Nao robot demo " width="480" height="360" border="0" /></a>


Nao robots are star players in RoboCup, an annual autonomous robot soccer competitions.
We are planning to build the Maryland RoboCup team to compete in RoboCup 2019, we need your help.
Would you like to help us in Nao's soccer training? We need to train Nao to detect a soccer ball and estimate the depth of the ball to know how far to kick.

Nao's training has two phases:
- Color Segmentation using Gaussian Mixture Model (GMM)
- Ball Distance Estimation

<a name='problem'></a>
## What you need to do
To make logistics easier, we have collected camera data from Nao robot on behalf of you and saved the data in the form of color images. Click [here](https://drive.google.com/file/d/17XiM86JqHqko4JC00-E4w4sPKnzh2iMz/view?usp=sharing) to download. The image names represent the depth of the ball from Nao robot in centimeters. **The test dataset is [here](https://drive.google.com/file/d/17tNn3YIVR-8kqoBgJNK58YY4UBnQmm4q/view?usp=sharing) to download**.

<a name='pro'></a>
### Problem Statement
0. Prepare the data: Extract the regions of the ball from each of the training images. For example, you can use the [*roipoly*](https://github.com/jdoepfert/roipoly.py) function to do so. Please note that since it is a Python script you would have to extract these features outside of the Jupyter notebook. The image points obtained this way are the data that you will use to train your color model.
1. Cluster the orange ball using a [Single Gaussian](https://nayeemmz.github.io/cmsc426fall2019/colorseg/#gaussian). You may use the Python code from previous homework. [30 points]
2. Cluster the orange ball using a [Gaussian Mixture Model](https://nayeemmz.github.io/cmsc426fall2019/colorseg/#gmm). You may use the Python code from the previous homework. [40 points] 
3. Estimate the [distance](https://nayeemmz.github.io/cmsc426fall2019/colorseg/#distest) to the ball [20 points]. 
4. Plot all the [GMM ellipsoids]( https://nayeemmz.github.io/cmsc426fall2019/assets/hw3/homework_hw3_draw_ellipsoid.ipynb) [10 points].

To help you with code implementation, we have given the pseudocode :-)


<div class="fig fighighlight">
  <img src="/cmsc426fall2019/assets/proj1/proj1_image.PNG" width="100%">
  <div class="figcaption">
  </div>
  <div style="clear:both;"></div>
</div>

<a name='sub'></a>
## Submission Guidelines


### File tree and naming

Your submission on Canvas must be a zip file, following the naming convention **YourDirectoryID_hw3.zip**.  For example, xyz123_hw3.zip.  The file **must have the following directory structure**.

YourDirectoryID_proj1.zip.
 - train_images/.
 - test_images/.
 - results/.
 - GMM.ipynb
 - trainGMM.ipynb
 - testGMM.ipynb
 - measureDepth.ipynb
 - plotGMM.ipynb
 - report.pdf

### Report

For each section of the project, explain briefly what you did, and describe any interesting problems you encountered and/or solutions you implemented.  You must include the following details in your writeup:

- Your choice of color space, initialization method and number of gaussians in the GMM.
- Explain why GMM is better than a single Gaussian.
- Present your distance estimate and cluster segmentation results for each test image.
- Explain strengths and limitations of your algorithm. Also, explain why the algorithm failed on some test images.

As usual, your report must be full English sentences, **not** commented code. There is a word limit of 1500 words and no minimum length requirement

<a name='coll'></a>
## Collaboration Policy
You are encouraged to discuss the ideas with your peers. However, the code should be your own, and should be the result of you exercising your own understanding of it. If you reference anyone else's code in writing your project, you must properly cite it in your code (in comments) and your writeup. For the full honor code refer to the CMSC426 Fall 2019 website.
