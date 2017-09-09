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

## Development Environment 
* Intel® Core™ i5-5200U CPU @ 2.20GHz × 4
* GeForce 930M/PCIe/SSE2
* Anaconda - Spyder (GPU Tensor Flow/Theano Backend)
* Linux Ubuntu 17.04


[//]: # (Image References)

[image1]: ./examples/nvidiaNN.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_30_48_287.jpg
 "Center"
[image3]: ./examples/left_2016_12_01_13_38_52_961.jpg
 "Left"
[image4]: ./examples/right_2016_12_01_13_33_34_260.jpg
 "Right"
[image5]: ./examples/loss.png "Loss without Generator"
[image6]: ./examples/loss_2.png "Loss with Generator"
[image7]: ./examples/loss_udata_gen.png "Loss Using Udacity DataSet"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model_gen.py containing the script to create and trains the model with a generator implemented 
* drive.py for driving the car in autonomous mode with PID controller 
* model.h5 containing a trained convolution neural network 
* video.mp4 Video of using "python drive.py model.h5" command
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. The NVidia AutoPilot Architecture.

The overall strategy was using the powerful NVidia Autopilot ConvNet. NVidia is one of the greatest researchers in Deep Learning, so will be very tricky to concept a Neural Network that performs better.

The model consists in a 9 layers network, 5 convolutionals layers , 3 fully connected and 1 normalization layer. I decided to use this ConvNet due the power to process images this network has. I just had to change the input for a 3@160X320 image.

![alt text][image1]

After normalizing using a Keras lambda layer the input (line 53) , the image was cropped: 70 pixels in the top and 25 in the bottom.

#### 2. Overfitting

Experiments using Dropout were done to prevent overfitting, but no improvement were observed in the performance as well in the loss, so the ConvNet was completely implemented as the original AutoPilot ConvNet from NVidia.

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .
In drive.py the PI controller was fine-tuned as well a Kd parameter and derivation error were included to prevent oscillations during the autonomous drive. The proportional gain as well the integrator were tuned to perform faster , but these improvements could change the behavior of the car in the track that could oscillate until loose control, this problem was compensated by the introdution of the Derivative Gain.
Turning the PI in a PID controller was very important to have a simple model with a modest data set.

Some test were done with large datasets using the same model but with a Python Generator (is included as model_gen.py). After hundred of experiments the model performs much better with a smaller dataset and a fine tuned PID controller.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center,left and right lane images. The angles measurements were offsetted by 0.12 in left and right images. This decision augmented the dataset by 3 times, even so the resulted dataset was duplicated again with a flipped image followed by the steer angle multiplied by -1.(lines 31 to 45).

The training dataset was recorded with two laps. One laps zig-zagging and another center lane driving. 

Center Image

![alt text][image2]

Left Image

![alt text][image3]

Right Image

![alt text][image4]

#### 5. Appropriate validation data

The validation data was splitted from the original trainning data in a proportion of 20%. To split the data the parameter Shuffle was turned active (True, line 73).

#### 6. Test and Validation Results 

The performance was compare within two models, one with generator and another without. The ConvNet was rigorous the same, the only difference was the way the memory was used. To train the model I used a NVidia 930 GPU, so was possible to fit entire model without a generator.
The final result with no generator model performed better than with generator as is possible to verify below:

Model without Generator

![alt text][image5]

Model with Generator

![alt text][image6]

#### 7. Conclusions 

The first interesting thing is that using the UDACITY provided dataset the loss was pretty much lower than with my recorded dataset, but curiously the driving performance , using the same drive.py and model_gen.py , was worst, actually, the car was not able to keep the track.

Loss with Udacity DataSet

![alt text][image7]

But as can be checked in the video.mp4 file, a combination of a well recorded DataSet with a PS4 joystick and a fine PID tunning was enough to keep the car driving stricly on track.











