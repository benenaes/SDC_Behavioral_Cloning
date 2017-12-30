---
typora-root-url: ./
---

# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behaviour
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submitted files

This project includes the following files:
* model.py containing the script to create and train the model 
* drive.py for driving the car in autonomous mode
* model-v16.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* /videos contains all the videos that were recorded by drive.py and converted to MPEG-4 format by video.py
* /data contains all the images and CSV files that were recorded for training purposes
* /models contains all the trained models who didn't make it in the end

#### 2. How to run the code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-v16.h5
```

### Training Data

All the training data that was recorded, can be found in subfolders of the *data* subfolder. I added a header manually to all generated CSV files (like in the original CSV from Udacity), so that data could be read out using the *DictReader* from the CSV library (less error prone than working with field indices). 

All the training/validation data is read and parsed by the *parse_log()* and *parse_all_data()* functions in **model.py** 

- *middle* : This is a recording of three laps on the "lake" track using centre lane driving. I tried to keep the speed constant at around 15 mph (in all recordings) to avoid that speed/throttle/braking would play a role in the steering angle.  Also the images of the left and right cameras are used. An offset to the steering angle of +0.10 and -0.10 is applied respectively.
  ![center_2017_12_21_22_27_38_011](.\writeup-images\center_2017_12_21_22_27_38_011.jpg)
- *left2* and *right2*: I needed recordings where the network would learn to recover from situations where the car wasn't following the middle of the road anymore. It was suggested in the training material to make a recovery track, but I decided to follow a different strategy. I drove two tracks where I was trying to follow the left- and the rightmost parts of the road. I parsed the data of these track recordings and added an offset to the steering angle (+0.25 and -0.25), so that the network is guided towards the centre of the road again. That way, there is a recording for a recovery at almost each point in the track. First off, I tried an even more extreme strategy (recorded in *left* and *right*), where I would drive on the very edge of the road. This had the disadvantage that there was too little data of the space in between the centre and the edge of the road. This resulted in networks that were driving straight off to the edge of the road at some points and only recovering very late. These were the only recordings where the left and right camera images were not used.
  ![center_2017_12_24_15_54_18_522](/writeup-images/center_2017_12_24_15_54_18_522.jpg)
- *turns*: An additional recording with only the sharper turns of the lake track to balance out all the recordings with steering angle close to 0.
  ![center_2017_12_24_00_41_21_697](/writeup-images/center_2017_12_24_00_41_21_697.jpg)
- *mountains* and *mountains_reverse*: Since the mountain track contains a lot of sharper turns (and also parts where less of the road is visible when driving up a hill), the network was performing very poorly on this track. These recordings (one way and in reverse) would make the car drive a lot better on the mountain track, although it is a pity that by adding this data to the training data, we cannot cross check anymore on how well the network generalizes on other (unseen) tracks.
  ![center_2017_12_24_15_29_25_579](/writeup-images/center_2017_12_24_15_29_25_579.jpg)
- *mountains_turns*: It was already mentioned that there are a lot of sharper turns in the mountain track. This recording focussed on the really sharp turns in the track, so that we would have more of this type of data to train with.
  ![center_2017_12_25_23_42_56_167](/writeup-images/center_2017_12_25_23_42_56_167.jpg)
- *mountains_turns2*: There is one particularly interesting part in the mountain track where there is a hairpin turn. I wanted this kind of data a bit better represented in the training data, so I added this one as well.
  ![center_2017_12_26_22_21_46_223](/writeup-images/center_2017_12_26_22_21_46_223.jpg)

### Data Balancing

This data combined resulted in the following histogram of steering angles. 20 bins were used.

![OrigDataHistogram](/writeup-images/OrigDataHistogram.jpg)

Total number of samples (image flips included): 117412 

It is clearly visible that - even with the extra efforts to add extra data from turns - there is a lot more data with steering angle around 0 degrees. Therefore, I decided to rebalance the histogram by throwing away some data that was over-represented in the data set. This is implemented in *prune_samples()* in **model.py**. First off, a target average per class (bin) was calculated (*avg_samples_per_bin*). If a bin contained more samples than this average, then images were discarded with a probability that is proportional to the size factor above this average. This resulted in the following histogram:

![RebalancedDataHistogram](/writeup-images/RebalancedDataHistogram.jpg) 

Total number of samples after rebalancing: 67114

### Data Augmentation

There was not a need for a lot of data augmentation, since:

- A lot of data is already provided. The data already contains different tracks with different light conditions (like e.g. the shadowy parts in the mountain track)
- The network is already performing well with the used data. Unfortunately, both tracks were used and I had no other way to validate how well the network generalizes, i.e. performs on unknown tracks.

All the data augmentation happens during the training of the track itself in the *image_generator()* generator function:

- Each image in the batch is read with *cv2.cvtColor()* and immediately converted from BGR to RGB colour space
- If the data recording defines that the image needs to be flipped, then *cv2.flip()* is used and the steering angle is reversed accordingly. 

There are couple of other image augmentations that definitely could be used to make the network generalize better:

- Randomly change the brightness of an image (but not too harshly)
- Perspective transformations around the X axis (not around the Y axis as this would have an influence on the ground truth of the steering angle)

Finally the data set is shuffled and split into a 80% part for the training data and 20% for the validation data. Of course, these data sets are also shuffled in the training phase by the *image_generator()* function.  

### Model Architecture and Training Strategy

#### 1. Model architecture

The used model is based on the NVidia model explained in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The final model consisted of the following layers - implemented by *build_model()* :

| Layer Description                        | Keras Layer Type                         |
| ---------------------------------------- | ---------------------------------------- |
| Cropping of the image from 320 x 160 to 320 x 55 (25 rows from the lowest part and 80 rows from the upper part). The upper part contains no useful information (sky, trees, mountains). Also, since this isn't a recurrent network, only close-by image data contains the most interesting information to determine steering angles. The aggressive cropping was one of the most important decision factors to make the network perform well. | Cropping2D with parameters (80,25), (0,0) |
| Normalization of the input data: all pixel data is linearly transformed to the [-1,1] range | Lambda                                   |
| Convolutional layer 1 with L2 regularization <br />Notice that there is 1 convolutional layer less than in the NVidia model due to the fact that the aggressive cropping doesn't allow an extra convolutional layer) | Conv2D with kernel size (5,5), strides (2,2), valid padding and filter depth 36. Weights L2 regularization factor: 0.001 |
| Optional dropout layer: not activated since it didn't improve training and validation loss | Dropout                                  |
| Convolutional layer 2                    | Conv2D with kernel size (5,5), strides (2,2), valid padding and filter depth 48. Weights L2 regularization factor: 0.001 |
| Optional dropout layer: not activated    | Dropout                                  |
| Convolutional layer 3                    | Conv2D with kernel size (3,3), strides (1,1), valid padding and filter depth 64. Weights L2 regularization factor: 0.001 |
| Optional dropout layer: not activated    | Dropout                                  |
| Convolutional layer 4                    | Conv2D with kernel size (3,3), strides (1,1), valid padding and filter depth 64. Weights L2 regularization factor: 0.001 |
| Flatten layer                            | Flatten                                  |
| Optional dropout layer: not activated    | Dropout                                  |
| Fully connected layer 1                  | Dense with 100 hidden nodes. Weights L2 regularization factor: 0.001 |
| Optional dropout layer: not activated    | Dropout                                  |
| Fully connected layer 2                  | Dense with 50 hidden nodes. Weights L2 regularization factor: 0.001 |
| Optional dropout layer: not activated    | Dropout                                  |
| Fully connected layer 3                  | Dense with 10 hidden nodes.              |
| Fully connected layer 4                  | Dense with 1 final node                  |
|                                          | ooooooooooooooooooooo                    |

All convolutional and fully connected layers have RELU activation functions to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

There was a lot of training/validation data and I kept the amount of epochs quite low. I didn't perceive that the training and validation mean square error started to diverge, so I decided to not activate the dropout layers in the model. 

#### 3. Model parameter tuning

- The model used an Adam optimizer with initial learning rate 0.0001, so the learning rate was not tuned manually.
- The batch size was 256. This was the maximum my GPU could cope with. This batch size is already big enough to be quite certain that each bin has enough entropy (and thus that gradient descent is not steered too much in a single direction).
- The L2 regularization factor for the weights was chosen at 0.001. I saw an improvement in the results by introducing this into the network: the car was driving far more stable. 
- Number of training epochs: 7. After that, the MSE of the validation set didn't decrease anymore. Weights are saved after each training epoch (with a *ModelCheckpoint* callback), just in case the training and validation loss would start to diverge.

#### 4. Training results

The following graph shows the evolution of the loss at the end of each epoch (zero indexed).

![Loss](/writeup-images/Loss.jpg)

It has to be noted that the loss for the training and validation set are close to each other at the end of epoch 7. This gives me the confidence that more training is not desired anymore (also tested empirically) and that the training and validation sets are both well balanced.

### Network performance

The model was saved in **model-v16.h5**. This model was used for the video recordings where the network was driving the car autonomously. 

**drive.py** was adapted due to the fact that the Windows simulator was using a different CUDA version. Since Keras uses the GPU version of Tensorflow (and this version uses an older CUDA version), this resulted in problems when getting the required cuBLAS and cuDNN handles. This was resolved by already performing a prediction with the trained network even before the simulator connects to the socket server **drive.py**. This way, the dynamic link library was directly loaded at start up of the Python script (and thus the right one was loaded). Another solution (commented out on lines 15-16) was provided to make Keras/Tensorflow run on CPU: this was also a valid solution, since the feed forward phase doesn't take that much CPU. 

The first experiment was on the lake track, where we fixed the speed at 15 mph. The results can be found in **videos/lake-15mph.mp4**

![2017_12_27_21_29_24_500](/writeup-images/2017_12_27_21_29_24_500.jpg)

The second experiment was on the same lake track, but now at the maximum speed of 30 mph. The results can be found in **videos/lake-30mph.mp4**

The third experiment was on the mountain track, at 15 mph. 

![2017_12_27_21_42_02_691](/writeup-images/2017_12_27_21_42_02_691.jpg)

I also tried the mountain track at 25 mph, but the car got stuck in the hairpin turn somewhere in the last part of the track. The driving was also quite nervous and would definitely make passengers sick. Room for improvement !