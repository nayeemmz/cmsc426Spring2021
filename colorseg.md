---
layout: page
mathjax: true
permalink: /colorseg/
---

Table of Contents:

- [Color Classification](#colorclassification)
	- [Color Classification using a Single Gaussian](#gaussian)
	- [Color Classification using a Gaussian Mixture Model (GMM)](#gmm)
  - [Different cases for $$\Sigma$$ in GMM](#gmmcases)


<a name='gaussian'></a>
### Clustering using a Single Gaussian
You may think of this problem as assigning a color label to each pixel. Each distribution may represents a red, green or blue cluster. We want to cluster a pixel(point) as red, green or blue as we have a total of three classifiers one for each color. Let us formulate the problem mathematically. In each classifier, we want to find $$p(C_l \vert x)$$. Here $$C_l$$ denotes the cluster label. So as you expect the first classifier will give you the following $$p(C_1 \vert x)$$, i.e., probability that the pixel is the first cluster or may be belong to the red channel. Note that $$1 - p(C_1 \vert x)$$ gives the probability that the pixel is not from cluster 1 which includes both cluster 2 and 3 points (pixels). 

Estimating $$p(C_l \vert x)$$ directly is too difficult. Luckily, we have Bayes rule to rescue us! Bayes rule applied onto $$p(C_l \vert x)$$ gives us the following:

$$
p(C_l \vert x) = \frac{p(x \vert C_l)p(C_l)}{\sum_{i=1}^k p(x\vert C_i)p(C_i)}
$$

$$p(C_l \vert x)$$ is the conditional probability of a cluster label given the cluster observation and is called the **Posterior**. $$p(x \vert C_l)$$ is the conditional probability of cluster observation given the cluster label and is generally called the **Likelihood**. $$p(C_l)$$ is the probability of that cluster occuring and is called the **Prior**. The prior is used to increase/decrease the probability of certain clusters. If nothing about the prior is known, a common choice is to use a uniform distribution, i.e., all the clusters are equally likely. This type of clustering is an unsupervised leaerning problem. Unsupervised means that we do not have "training" examples from which we can understand the cluster we are looking for. 

For the purpose of easy discussion, we want to look for the points that are similae to each other and are more likely to come from the same distribution and hence grouped together. The Likelihood is generally modelled as a normal/gaussian distribution given by the following equation:

$$
p(x \vert Orange) = \frac{1}{\sqrt{(2 \pi)^3 \vert \Sigma \vert}}\exp{(\frac{-1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))} = \mathcal{N(x \vert \mu, \Sigma)}
$$

Here, $$\vert \Sigma \vert$$ denotes the determinant of the matrix $$\Sigma$$. The dimensions of the above terms are as follows: $$\Sigma \in \mathbb{R}^{3 \times 3}, x,\mu \in \mathbb{3 \times 1}, p(x \vert Orange) \in \mathbb{R}^1$$. 

You might be asking why we used a Gaussian distribution to model the likelihood. The answer is simple, when you average a lot of (theoretically $$\infty$$) independently identically distributed random samples, their distribution tends to become a gaussian. This is formally called the [**Central Limit Theorem**](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/sampling-distribution-mean/v/central-limit-theorem). 

All the math explanation is cool but how do we implement this? It's simpler than you think. All you have to do is find the mean ($$\mu$$) and covariance ($$\Sigma$$) of the likelihood gaussian distribution. Let us assume that we have $$N$$ samples for the class 'Orange' where each sample is of size $$\mathbb{R}^{3 \times 1}$$ representing the red, green and blue channel information at a particular pixel. The empirical mean $$\mu$$ is computed as follows:

$$
\mu = \frac{1}{N}\sum_{i=1}^N x_i
$$

here $$i$$ denotes the sample number. The empirical co-variance $$\Sigma$$ is compted as follows:

$$
\Sigma = \frac{1}{N}\sum_{i=1}^N (x_i-\mu)(x_i-\mu)^T
$$

Clearly, $$\mu \in \mathbb{3 \times 1}$$ and $$\Sigma \in \mathbb{R}^{3 \times 3}$$. $$\Sigma$$ is an awesome matrix and has some cool properties. Let us discuss a few of them.  

The co-variance matrix $$\Sigma$$ is a square matrix of size $$d \times d$$ where $$d$$ is the length of the vector $$x$$, i.e., $$\Sigma \in \mathbb{R}^{d \times d}$$ if $$x \in \mathbb{R}^{d \times 1}$$. For the RGB case, $$d=3$$. $$\Sigma$$'s diagonal terms denote the variance and the off-diagonal terms denote the correlation. Let us take the example of the RGB case. If $$x = [R, G, B]^T$$, then 

$$
\Sigma = \begin{bmatrix}
\sigma_R^2 & \sigma_R \sigma_G & \sigma_R \sigma_B \\
\sigma_R \sigma_G & \sigma_G^2 & \sigma_G \sigma_B \\
\sigma_R \sigma_B & \sigma_G \sigma_B & \sigma_B^2 \\
\end{bmatrix}
$$

Observe that the above matrix is a **square** matrix and is a **symmetric** matrix. Here, $$\sigma_R, \sigma_G, \sigma_B$$ denote the variance in each of the individual channels. $$\sigma_R^2, \sigma_G^2, \sigma_B^2$$ are the variance in each of the R, G and B channels. $$\sigma_R \sigma_G, \sigma_G \sigma_B, \sigma_R \sigma_B$$ are the correlation terms and show the co-occurence of one channel over other. Mathematically, it signifies the vector projection of one channel over the other.  

An important property to know about $$\Sigma$$ is that it is a **Positive Semi-Definite (PSD)** Matrix and is denoted mathematically as $$\Sigma \succeq 0$$. This means that the [eigenvalues](http://mathworld.wolfram.com/Eigenvalue.html) are non-negative (either positive or zero). This physically means that you cannot have a negative semi-axes for the ellipse/elliposoid which makes sense. The [eigenvectors ](http://mathworld.wolfram.com/Eigenvector.html) of $$\Sigma$$ tell you the orientation of the elliposoid in 3D. A function [like this](https://www.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse?focused=3890020&tab=function) can help you plot the covariance ellipsoids. 

Now that we have both the prior and likelihood defined we can find the posterior easily:

$$
p(C_l \vert x) = \frac{p(x \vert C_l)p(C_l)}{\sum_{i=1}^l p(x\vert C_i)p(C_i)}
$$

Because we just want to find the colors by some thresholding later, we can drop the denominator in the above expression if we don't care about the absolute scale of the probability summing to 1. For most thresholding purposes, we can do the following approximation:

$$
p(C_l \vert x) \propto p(x \vert C_l)p(C_l)
$$

So using the following expression one can identify pixels which are 'Orange' (or confidently Orange). 

$$
p(C_l \vert x) \ge \tau
$$

Here, $$\tau$$ is a user chosen threshold which signifies the confidence score. This method definitely works much better than the [simpler color thresholding method](#colorthresh). All your data is being thresholded by an ellipsoid (3D ellipse) instead of a cube as before. You might be wondering why a gaussian looks like an ellipsoid? The covariance matrix represents the semi-axes of the ellipsoid. In fact the inverse of square root of diagonal values of $$\Sigma$$ gives the semi-axes of the ellipsoid. As you would expect if $$x \in \mathbb{R}^{2 \times 1}$$, the gaussian would look like an ellipse. Learn more about these cool gaussians [here](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).

Modelling the likelihood as a gaussian is beneficial because a little light variation generally makes the colors spread out in an ellipsoid form, i.e., the actual color is in the middle and color deviates from the center in all directions resembling an ellipse. This is one of the major reasons why a simple gaussian model works so well for color segmentation. 

<a name='gmm'></a>
### Color Classification using a Gaussian Mixture Model (GMM)
However, if you are trying to find a color in different lighting conditions a simple gaussian model will not suffice because the colors may not be bounded well by an ellipsoid. 

<div class="fig figcenter fighighlight">
  <img src="/assets/colorseg/ballindiffcolorspaces.png">
  <div class="figcaption">Datapoints for the ball plotted in RGB and YCbCr colorspaces. Observe how the enclosing shape is of a weird shape. It would be ideal to create a custom color-space which converts this weird shape into some simple shape which can be enclosed like a cube or a cuboid or a sphere or an ellipsoid. Designing a space like that is generally not trivial hence we emply a method of fitting this weird shape as a sum of simple shapes like an ellipsoid.</div>
</div>

In this case, one has to come up with a weird looking fancy function to bound the color which is generally mathematically very difficult and computationally very expensive. An easy trick mathematicians like to do in such cases (which is generally a very good approximation with less compuational cost) is to represent the fancy function as a sum of known simple functions. We love gaussians so let us use a sum of gaussians to model our fancy function. Let us write our formulation down mathematically. Let the posterior be defined by a sum of $$K$$ scaled gaussians given by:

$$
p(C_l \vert x) = \sum_{i=1}^k \pi_i \mathcal{N}(x, \mu_i, \Sigma_i)
$$

Here, $$\pi_i$$, $$\mu_i$$ and $$\Sigma_i$$ respectively define the scaling factor, mean and co-variance of the $$k$$<sup>th</sup> gaussian. The optimization problem in hand is to maximize the probability that the above model is correct, i.e., to find the parameters $$\pi_k, \mu_k, \Sigma_k$$ such that one would maximize the corectness of $$p(C_l \vert x)$$. Just a simple probability function doesnt have very pretty mathematical properties. So a general trick mathematicians/machine learning people follow is to take the logarithm of the probability function and maximize that. This works well because of the [monotonicity](http://mathworld.wolfram.com/MonotonicFunction.html) of the logarithm function. This setup is formally called **Maximum Likelihood Estimation (MLE)** and can be mathematically written as:

$$
\underset{\{ \mu_1, \mu_2, \cdots, \mu_k, \Sigma_1, \Sigma_2, \cdots, \Sigma_k, \pi_1, \pi_2, \cdots, \pi_k\}}{\operatorname{argmax}} \sum_{i=1}^N \log p(x_i)
$$

where $$N$$ is the number of training samples. The above is not a simple function and generally has no closed form solution. To solve for the parameters $$\Theta = \{ \mu_1, \mu_2, \cdots, \mu_k, \Sigma_1, \Sigma_2, \cdots, \Sigma_k, \pi_1, \pi_2, \cdots, \pi_k\}$$ of the above problem, we have to use an iterative procedure. 

- Initialization:
Randomly choose $$\pi_i, \mu_i, \Sigma_i \qquad \forall i \in [1, k]$$
- Alternate until convergence:
	- Expectation Step or E-step: Evaluate the model/Assign points to clusters
	Cluster Weight $$ \alpha_{i,j} = \frac{\pi_i p(x_j \vert C_i)}{\sum_{i=1}^k \pi_i p(x_j \vert C_i)} $$
	\\(j\\) is the data point index, \\(i\\) is the cluster index.
	- Maximization Step or M-step: Evaluate best parameters $$ \Theta $$ to best fit the points
	
	$$ 
	\mu_i = \frac{\sum_{j=1}^N \alpha_{i,j} x_j}{\sum_{j=1}^N \alpha_{i,j}}
	$$
	

	$$ 
	\Sigma_k = \frac{\sum_{j=1}^N \alpha_{i,j} (x_j-\mu_i)(x_j-\mu_i)^T}{\sum_{j=1}^N \alpha_{i,j}}
	$$

	$$ 
	\pi_i = \frac{1}{N}\sum_j \alpha_{i,j}
	$$
	
Convergence is defined as $$\sum_i\vert \vert \mu_i^{t+1} -  \mu_i^{t}\vert \vert \le \tau$$ where $$i$$ denotes the cluster number, $$t$$ denotes the iteration number and $$\tau$$ is some user defiened threshold. To understand more about the mathematical derivation which is fairly involved go to [this link](https://alliance.seas.upenn.edu/~cis520/dynamic/2017/wiki/index.php?n=Lectures.EM).

Now that we have estimated/learnt all the parameters in our model, i.e., $$\Theta = \{ \mu_1, \mu_2, \cdots, \mu_k, \Sigma_1, \Sigma_2, \cdots, \Sigma_k, \pi_1, \pi_2, \cdots, \pi_k\}$$ we can estimate the posterior probability using the following equation:

$$
p(C_l \vert x) = \sum_{i=1}^k \pi_i \mathcal{N}(x, \mu_i, \Sigma_i)
$$ 

Finally, one can use the following expression to identify pixels which are 'Orange' (or confidently Orange). 

$$
p(C_l \vert x) \ge \tau
$$

here $$\tau$$ is some user defined threshold. 

<a name='gmmcases'></a>
### Different cases for $$\Sigma$$ in GMM
We said that we were modelling our fancy functions as a sum of simple functions like a gaussian. One might wonder why cant we make further asumptions about the gaussian. Yes we can! The gaussian we described before uses an ellipsoid, i.e., all the diagonal elements of $$\Sigma$$ are different. One can say that all our diagonal elements are the same and non-diagonal elements are zero, i.e., $$\Sigma$$ has the following form:

$$
\Sigma = \sigma^2\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
\end{bmatrix}
$$


here $$\sigma$$ is a parameter to be estimated. You might be wondering what shape a $$\Sigma$$ like the one described above represents. It's simple, a sphere! This gives lesser flexibility in fitting the model as the shape is simpler but has lesser number of parameters. A comparison of GMM fit on the data using spherical $$\Sigma$$ and elliposoidal $$\Sigma$$ is shown below:


<div class="fig figcenter fighighlight">
  <img src="/assets/colorseg/rgbgmm.png">
  <div class="figcaption">Left: Datapoints of the orange ball and GMM fit using spherical \(\Sigma\). Right: GMM fit using ellipsoidal \(\Sigma\). Notice that the ellipsoidal variant has less non-orange pixels which will be classified as orange, i.e., lesser false positives and false negatives and is more accurate.</div>
</div>

One might think, what if I design a custom transformation to create a new colorspace from RGB where the data points are enclosed in a simple shape like an ellipsoid? That would work wonderfully well. I designed a custom colorspace to do exactly that (which is my secret recipe). You will have to figure out your own secret recipe to do it. The datapoints and the GMM fit for this colorspace is shown below:

<div class="fig figcenter fighighlight">
  <img src="/assets/colorseg/customcolorspace3.png">
  <div class="figcaption">Left: Datapoints of the orange ball in the custom colorspace. Look at how the datapoint space looks like an ellipsoid Right: GMM fit using ellipsoidal \(\Sigma\). Notice that the GMM fit looks exactly like one single gaussian which shows that the performance of GMM over this colorspace would exactly be the same as a single gaussian fit. This is very beneficial because we can reduce the computation cost of training and testing significantly.</div>
</div>



<!-- When git doesn't push do this: git config --global core.askpass "git-gui--askpass" -->
