# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidiaNN.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model_gen.py containing the script to create and trains the model with a generator implemented 
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. The NVidia AutoPilot Architeture.

The model consists in a 9 layers network, 5 convolutionals layers , 3 fully connected and 1 normalization layer. I decided to use this ConvNet due the power to process images this network has, I just had to change the input for a 3@160X320 image.

![alt text][image1]

After normalizing using a Keras lambda layer the input (line 53) , the image was cropped: 70 pixels in the top and 25 in the bottom.

#### 2. Overfitting

Experiments using Dropout were done to prevent overfitting, but no improvement were observed in the performance as well in the loss, so the ConvNet was completely implemented as the original AutoPilot ConvNet from NVidia.

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .
In drive.py the PI controller was fine-tunned as well a Kd parameter and derivation error were included to prevent oscilations during the autonomous drive. The proportional gain as well the integrator were tunned to perform faster , but these improvements could change the behavior of the car in the track that coud oscilate until loose control, this problem was conpensated by the introdution of the Derivative Gain.
Turning the PI in a PID controller was very important to have a simple model with a modest data set.

Some test were done with large datasets using the same model but with a Python Generator (is included as model_gen.py). After hundred of experiments the model performs much better with a smaller dataset and a fine tunned PID controler.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center,left and right lane images. The angles measurements were offsetted by 0.12 in left and right images. This decision augmented the dataset by 3 times, even so the resulted dataset was duplicated again with a flipped image followed by the steer angle multiplied by -1.(lines 31 to 45)

#### 5. Appropriate validation data

The validation data was splitted from the orginal trainnning data in a proportion of 20%. To split the data the parameter Shuffle was turned active (True, line 73).



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was using the powerful NVidia Autopilot ConvNet. NVidia is one of the greatest researchers in Deep Learning, so will be very tricky to concept a Neural Network that performs better.


Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
