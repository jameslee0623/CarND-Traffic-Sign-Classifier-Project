# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_img/04.jpg "Traffic Sign 4"
[image5]: ./new_img/12.jpg "Traffic Sign 12"
[image6]: ./new_img/13.jpg "Traffic Sign 13"
[image7]: ./new_img/14.jpg "Traffic Sign 14"
[image8]: ./new_img/17.jpg "Traffic Sign 17"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And Traffic_Sign_Classifier.ipynb is my project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 34799
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how different classes distribute. From the chart, you can see we have at least around 200 images for each class. And at most around 2000 for some other classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Although all the articles I read do the grayscale at first, I'm still thinking by doing grayscale, we only lost some data instead of gaining. So I decide NOT to do the grayscle at this point.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

So the only step of preprocessing I do is normalization. 
I simply do ```python x_norm = (x - 128)/128``` at the first try. But I found out it might make my image data int between 0 and 1, a.k.a. only 0 or 1. So I changed it to 
```python 
cv2.normalize(X_train, X_train_norm, -1, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
```
But function ```imshow``` can olny draw float image value between 0 to 1. So there's no way to visualization. But I did check the mean of image became -0.66 from 43.88, which is match the result from formula ``python x_norm = (x - 128)/128```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
|						|												|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
|						|												|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 1x1x800	|
| RELU					|												|
|						|												|
| Flatten layer			| flatten(conv2)5x5x32 + flatten(conv3)1x1x800	|
| Dropout 				| keep_prob = 0.5								|
| Fully connected		| Input = 1600. Output = 43.					|
| Softmax				| tf.nn.softmax_cross_entropy_with_logits()		|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The final model is above. Optimizer is below. 
```python
rate = 0.001
logits = LeNet(x, k_features)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
```python
EPOCHS = 80
BATCH_SIZE = 128
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 
* validation set accuracy of ? 0.960
* test set accuracy of ? 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet
* What were some problems with the initial architecture? Because there's no dropout at LeNet, it will overfitting while epochs goes big. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. Therefore, I adapted from Sermanet/LeCunn traffic sign classification journal article.
* Which parameters were tuned? How were they adjusted and why? I tried to save more features from the first and second layer of convolution.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Convolution helps us to find out the features around the area. And dropout helps us to prevent overfitting.

If a well known architecture was chosen:
* What architecture was chosen? The one from Sermanet/LeCunn traffic sign classification journal article.
* Why did you believe it would be relevant to the traffic sign application? It's designed for the problem.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? I got around 0.96 accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

It has a 100% correct rate.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.5%. Due to the images I found are brighter and clearer. It's easier to the model to predicting.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Softmax: 
TopKV2(values=array(
[[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00],
 [  1.00000000e+00,   1.08178799e-26,   1.70834447e-33, 0.00000000e+00,   0.00000000e+00],
 [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00],
 [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00],
 [  1.00000000e+00,   5.05504256e-26,   8.18293642e-37, 2.80456996e-37,   0.00000000e+00]], dtype=float32), indices=array(
      [[ 4,  0,  1,  2,  3],
       [13, 28, 35,  0,  1],
       [14,  0,  1,  2,  3],
       [12,  0,  1,  2,  3],
       [17, 12, 38, 10,  0]], dtype=int32))

Correct answer: [4, 13, 14, 12, 17]

From the Softmax array and Correct answer above, the model is almost 100% certain for its predictions. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


