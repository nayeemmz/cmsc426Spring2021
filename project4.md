---
layout: page
mathjax: true
title: Implementation of Convolutional Neural Networks
permalink: /proj/p4/
---

Table of Contents:
- [Deadline](#due)
- [Introduction](#intro)
- [Implementation Overview](#system_overview)
- [Submission Guidelines](#sub)
- [Collaboration Policy](#coll)

<a name='due'></a>
## Deadline
11:59 PM, May 20, 2020

<a name='intro'></a>
## Introduction
In this project you will implement a Convolutional Neural Network (CNN) in two different ways: 
  * a step by step approach using regular Python to perform SIGNS data classification, and  
  * using Tensorflow framework to perform classification of MNIST digits dataset.
 
The goal of this assignment is to help you understand CNN's by building their different components and to carry out classification tasks. You will be applying your CNN's on the SIGNS and MNIST digits datasets. Although, you are being asked to use the SIGNS dataset in the first approach and the MNIST dataset for the second, you could try both these datasets using both approaches. However, please not that running the first approach on MNIST dataset may be very slow. In both approaches some of the components include, forward convolution, backward convolution(optional), zero padding, max-pooling and average-pooling. You will not be graded for the backpropagation code.

In order to help you implement this you are provided with [starter code](/cmsc426Spring2020/assets/proj4/proj4-starterFiles.zip) that contains three Jupyter notebook skeletal files and other required supplementary files and images necessary for this project. The files <i>cnn.ipynb</i> and <i>cnn-with-backprop.ipynb</i> are to be used for the step by step approach and the file <i>mnist_cnn.ipynb</i> is to be used for the Tensorflow framework approach. The descriptions of these files are as follows:

<ul>
  <li>cnn.ipynb - backpropagation algorithm is <b>not</b> implemented in this file. If you would like to give it a try, then start with this file for the step by step approach.
  </li>
  <li> cnn-with-backprop.ipynb - backpropagation algorithm is implemented in this file. Start with this file, if you don't want to try to implement it yourself.
  </li>
  <li> mnist_cnn.ipynb is to be used for our second approach using Tensorflow framework.
</ul>


<b> Note:</b> It is recommended that you start with cnn.ipynb file first and if you struggle with writing backprop code, then only use the cnn-with-backprop.ipynb file.


<a name='system_overview'></a>
## What to Implement

Most of the implementation details are provided to you in the Jupyter notebooks. You would be required to write code in these files identified by the comments in them. Each location is also identified with the number of lines of code that would be needed for a particular operation, along with the expected output of each cell. This would help you debug the code.


<a name='sub'></a>
## Submission Guidelines
Just one file, cnn.ipynb would be required to be submitted for this project.

Please add a new section before section 1 in the Jupyter notebook and include
 - The name of each group member.
 - A brief (one paragraph or less) description of what each group member contributed to the project.

<a name='coll'></a>
## Collaboration Policy
We encourage you to work closely with your groupmates, including collaborating on writing code.  With students outside your group, you may discuss methods and ideas but may not share code.

For the full collaboration policy, including guidelines on citations and limitations on using online resources, see <a href="http://www.cs.umd.edu/class/Spring2020/cmsc426-0201/">the course website</a>.
