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
  * a step by step approach using regular Python, and  
  * using Tensorflow framework to perform classification of MNIST digits dataset.
 
The goal of this assignment is to help you understand CNN's by building their different components. You will be applying your Tensorflow CNN implementation on the MNIST digits dataset classification. In both approaches some of the components include, forward convolution, backward convolution, zero padding, max-pooling and average-pooling. You will not be graded for the backpropagation code.

In order to help you implement this you are provided with [starter code](/cmsc426Spring2020/assets/proj4/proj4-starterFiles.zip) that contains three Jupyter notebook skeletal files and other required supplementary files and images necessary for this project. The files <i>cnn.ipynb</i> and <i>cnn-with-backprop.ipynb</i> are to be used for the step by step approach and the file <i>mnist_cnn.ipynb</i> is to be used for the Tensorflow framework approach. The descriptions of these files are as follows:

<ul>
  <li>cnn.ipynb - backpropagation algorithm is <b>not</b> implemented in this file. If you would like to give it a try, then start with this file for the step by step approach.
  </li>
  <li> cnn-with-backprop.ipynb - backpropagation algorithm is implemented in this file. Start with this file, if you don't want to try to implement it yourself.
  </li>
  <li> mnist_cnn.ipynb is to be used for our second approach using Tensorflow framework.
 </li>
</ul>


<b> Note:</b> It is recommended that you start with cnn.ipynb file first and if you struggle with writing backpropagation code, then only use the cnn-with-backprop.ipynb file.

A detailed description of these files is being skipped here because an elaborate documentation has been included in each one of these files. The comments in the files are self explanatory and include locations where you are required to fill in your code. In addtion, you may refer to the Artificial Neural Networks and Convolutional Neural Networks lectures covered in class.


<a name='system_overview'></a>
## What to Implement

Most of the implementation details are provided to you in the Jupyter notebooks. You would be required to write code in these files identified by the comments in them. Each location is also identified with the number of lines of code that would be needed for a particular operation. 


<a name='sub'></a>
## Submission Guidelines
You are required to submit the following files:
 * cnn.ipynb for the step by step approach to build a Convolutional Neural Network. It should contain your code filled along with the backpropagation algorithm that is provided to you. (40 points)
 * mnist_cnn.ipynb for the Tensorflow framework approach to classify MNIST digits data classification. (40 points)
 * report.pdf should contain a detailed description of the learnings from this project, difficulties you faced and the factors that impact classification accuracy of the test dataset. You should include a list of the parameters that you think need to be fine tuned. (20 points)
 


Please add a new markdown cell at the top of your Jupyter notebooks and include
 - The name of each group member.
 - A brief (one paragraph or less) description of what each group member contributed to the project.

<a name='coll'></a>
## Collaboration Policy
We encourage you to work closely with your groupmates, including collaborating on writing code.  With students outside your group, you may discuss methods and ideas but may not share code.

For the full collaboration policy, including guidelines on citations and limitations on using online resources, see <a href="http://www.cs.umd.edu/class/Spring2020/cmsc426-0201/">the course website</a>.
