# News-Image-Classification
This project trys to classify whether certain events occured in a news image. The images are labeled with whether something occured in them, such as violence, fire and a protest. An image can have several labels if a scene contains both a protest and violence,  and it is possible an image has no label if it is not related to any events of interest to be labeled.
There are 40,000 images in the dataset and 32,000 for model training and 8,000 for model testing.

### Model
There are several popular neural network model for computer vision applications such as ResNet and VGG. VGG model is a commonly used convolutional neural network model. Usually, it consists of 19 layers and it can be deeper. However, increasing network depth does not work by simply stacking layers together. Deep networks are hard to train because of the notorious vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly. The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers and this feature makes ResNet unique because it can guarantee performance and low complexity at the same time. So ResNet model is used in this project.

Several key arguments have to be taken care of is data_dir and cuda, data_dir identify the folder where datasets and annotation files store. Argument cuda identify if you want to accelerate the training precess using gpu computation units. Some analysis and evaluation plots are listed below.
### ROC curve for Protest
![]()
### Scatter plot for violence

### ROC curve for the rest labels
