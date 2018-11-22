# **Behavioral Cloning** 


[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/network_architecture.jpg "NVIDIA Architecture"
[image2]: ./examples/dirt_road.gif "Dirt Road GIF"
[image3]: ./examples/recover.gif "Recover GIF"
[image4]: ./examples/track2.gif "Track 2 GIF"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_t1.h5 containing a trained convolution neural network for track 1
* model_t2.h5 containing a trained convolution neural network for track 2
* this report for summarizing the results

The simulator that this project runs on can be found on [Udacity's GitHub](https://github.com/udacity/self-driving-car-sim).

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around track 1 by executing

```
python drive.py model_t1.h5
```

and driven autonomously around track 2 by executing

```
python drive.py model_t2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network designed for track 1 along with the ability to change a few lines of code to train the model for track 2 or even both tracks together. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose to use the [NVIDIA Architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for this project. This architecture consists of three convolution layers with 5x5 filter sizes and depths of 24, 36, and 48. Next there are two additional convolutional layers with 3x3 filter sizes and depths of 64 for both layers. Finally, the model is flattened and passed through four fully connected layers with output depths of 100, 50, 10 and 1. (model.py lines 115-124) 

The model includes RELU layers to introduce nonlinearity (code lines 115-119), and the data is normalized from 0 to 2 and mean centered from -1 to 1 in the model using a Keras lambda layer (code line 111). 

#### 2. Attempts to reduce overfitting in the model

The model reduced overfitting by training on a larger dataset (2,566 images for track 1 and 4,152 images for track 2) and by using a simple architecture that generalizes well. The images were then augmented to include a horiztanal mirror of the image to prevent bias toward left/right curves and a variation of +/- 0.2 was added to the steering measurement values to increase the window of accurate predicitons of the model, resulting in smoother turns. The large amount of data combined with the simple NVIDIA architecture allowed the model to perform well with minimal overfitting without the use of regularization such as dropout.

The model was trained and validated on different data sets to ensure that the model was not underfitting or overfitting (code lines 131-135). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and additional data in sections of the track that were different than that rest of the track such as the curve with one side being a dirt road in track 1.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with the NVIDIA CNN architecture and modify the network to get the desired results for this project. I thought this model was appropriate because it was designed for self-driving cars and was fairly lightweight so I felt that it would work well for my application.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. After adding image augmentation that trained the network on a horizontal mirror of the track and introducing steering measurement "wiggle", cropping the top region from the images, and normalizing and mean centering the images to values between -1 and 1, the model performed very well. 

The final step was to run the simulator to see how well the car was driving around track one. There was a spot where the vehicle failed to turn and drove into the dirt road portion of the track, to improve the driving behavior in these cases, I added more training data of the curve as shown below.


![alt text][image2]


Once track 1 was working, I tested the model on track 2 and it didn't stay on the road very well. To solve this problem I simply altered the batch size of the generators. Track 1 had a batch size of 128 because there were no sudden curves and the track was very consistent. This allowed the CNN's weights to be trained over larger sections of the track to help smooth out driving through curves. Track 2 had a much smaller batch size of 16. This allowed the CNN to perform well through the highly dynamic course without adjusting weights based on opposing curves in the track. (model.py lines 83-97)

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted  of three convolution neural networks with with 5x5 filter sizes and depths of 24, 36, and 48. Next there are two additional convolutional neural networks with 3x3 filter sizes and depths of 64 for both layers. Finally, the model is flattened and passed through four fully connected layers with output depths of 100, 50, 10 and 1. (model.py lines 115-124) 

This model was based on NVIDIA's CNN architecture and the following visualization is from their paper: "[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)"

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center of the lane if it veers to one side. These images show what a recovery looks like starting from the right side:

![alt text][image3]

Then I recorded any sections of the track that were difficult for the first iteration of the model. I then repeated this process on track two in order to get more data points.

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would would help the model generalize and not be biased toward left/right curves.

Another way I augmented the data was introducing "wiggle" into the steering measurements. This helped create a margin of error within +/- 0.2 of the actual steering measurement. Now the CNN can predict the correct steering angle with a much higher accuracy within the acceptable window.

After the collection process, I had 2,566 data points for track 1 and 4,152 data points for track 2. I then preprocessed this data by cropping the top of the image to remove the sky and then applied a lambda layer that adjusted the pixel values between 0 and 2. Next, I mean centered the images by subtracting 1.0 to move the pixel value range to -1 and 1.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was one for track 1 and two for track 2. I used an adam optimizer so that manually training the learning rate wasn't necessary. Track 1 had a training loss of 0.0305 and a validation loss of 0.0513 on epoch 1 while track 2 had a training loss of 0.0395 and a validation loss of 0.1396 on epoch 2.
