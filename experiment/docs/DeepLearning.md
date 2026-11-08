# Deep Feedforward Network

Learning patterns with the help of nurons alike mathematical models,which computes in layers,
feeds information to the forward layers to learn patterns from the dataset.

Composed result of many functions working together, with acyclic graph based flow.
target to best approximate the underlying pattern $ f^* $ function.

Nural network, is acyclic graph of functions which works in layers, to train parameters
for best approximating the underlying pattern in given dataset.FeedForward neural network
works with forwarding information flow with intermediate computations,
when computations are re-fed to the network its **recurrent nural network**.

## 1.1 Emergence of idea of nural networks

We can set ground with understanding scope of linear models, they work really well
with linear space and linear hyperplanes, with non-linear dataset we need to fix them, 
take them to the same structure to be able to use the linear functions.

With $\phi(x)$ being the mapping(rbf) to map features to infinite dimensional space for extracting the linear relations, such that we can apply the linear function.

**How to get the mapping $\phi$**
1. Using generic mapping functions

    + We can generalize well with training set
    + Testing set cant be easily generalized
    + focused on local smoothness of function
    + Non-encoding of enough prior information
2. Manual Engineer the mapping function

    + Needs domain specific skills
    + Required dedicated long term time
    + deligative task to perform

3. **Lock targets to learn**

    Being highly generic and specific with learning the parameter, <br>
    with training we get the specific maping functions out of all generic functions options.


Example : Learning **XOR** function,
with simple XOR maping we use non-linear function to map the input set into specific space for linear model to work with,
to make proper maping of XOR subset of information.

using non-linear function to describe features, With most nural networks does with, 
affine transformation through learned parameters,
with fixed non-linear activation functions called as activation function.

with the need of non-linear activation function in between we transform the feature space into a space where linear model could map out to the target well then before
We use non-linear functions because they could help to capture non-linear functions, help to map out to relevent spaces for linear function to workwith, 

Using multiple linear function in layers simply means using one due to mathematical property of collapsing into one linear function equivalency.

## 1.2 Gradient-Based Learning

## More reading
+ [Gradient Based learning | 172 PGN ](https://www.deeplearningbook.org/contents/mlp.html)

## Deep Learning with computer vision power 🤖

A branch where we deal with images to extract data and patterns thorugh numerical connection of images,
images we percieve as visual input, computers percieve with numerical values.

**Image as data** : Each image is a combination of multiple channels of matrix, filled with numbers,
each number in the matrix position tells the colour value at that position, forming the image.
We can extract those numbers through simple interfaces which converts the image format into standard numerical format.

With existing broad 3 classification with image data types,
+ Binary Images : with just two color, could be obtained with ReLu filter into normal images,
to convert them through one more filter of converting positive number to one,

+ Gray scale : with shades of gray, black and white, ranges through one colouring channel, 
represents good details of lights and shadows with the image.

+ Colourfull images : Loads with 3 channel of matrix with red blue and green channels, 
representing whole spectrum of images.

**Image tasks**
+ Object detection
+ Sematic segmentation
+ classification

### Using multilayer perceptron for image tasks

[!Note]
Due to requirement with flattening images, leading for very terrible learning improvements, 
due to severe flattening based loss of information through a image.

Naive Approach
1. Flattening makes distortion with relative positional spatial information of image
2. Massive trainable parameter requirements, really hard to scale up.
3. heavy transational sensitivity, cant learn positional details of images

## Convolutional network

Preserves the spatial information, works with relative positional information form image,
with its component coputationaly feasible with strong learning outcomes.

**Basic Terminologies**
+ Window : A fixed grid of input pixel which which represents the fundamental source of information for the whole network
+ Filter : Defined weights for a layer which takes element wise multiplication with its input layer to produce feature map for that input set of numbers
+ Feature Map : A ooutput set of preceding filters calculations, it is combinational result of multiple filters working together to produce a one section of feature map
+ receptive feild : defined per neuron, how much area in input image, pixel combination affects its activation. those perticular set of pixels are termed as receptive feild for the 
+ stride : A simple step size for shifting window to a direction requirement.
+ padding : Extra pixels added to the edges of the image 

[!Note]
Different Filter capture different features from the image how ?,
Due to masking at each filter, one filter would block out other section for that convolution, 
preventing from other non-relative features to sneek into the main frame,
this simple step makes the convolution network filters extracting differnet features at each injection.

### Components of network

Components of convolutional network provides robust way for computing pattern through out the image,
with combined with different techniques with the network, cnn serves robust architechture for serving image based model at scale.

1. Input
Network takes input through a defined size of pixel, Window element from the input image,
those picked pixels are converted into numerical representation for further layers

2. Filter 
small set of learnable weights, usually of same size as input window, also termed as kernel,
which does element wise multiplication sum to produce its feature maping value.

3. Feature map
A collection of values computed through different filter kernels throuh input windows,
this serves into hidden layer for more hierchy of knowledge throughout.

**Brief overview of whole convolutional flow**
A window of pixel is defined into the main image pixels, with defined stride and padding as per problem scope, the main window slides through the image, since due to masking rules one window captures only a portion of image thus only captures local features at first layer, <br>
With random initiated weights based kernel matrix which maps the input set into one number through element wise multiplication sum, thus the next hiddenn layer is connected to the main input layer,

With aggregated features maping output the next layer can proceed with the same sequence for much more dense layers, completing a usual forward pass.

[!Info] : This statement needs more clarity
All of the predictions are compared with the original target, loss is computed then backpropagation traces back the error within the network to adjust the weights and next cycle begins,

**Receptive domination magic**
with more layers into the CNN layer at each increased layer level network learns more abstract feature throughh the image, due to aggregated network information from previous layers, since the layers are all fully connected thus next layer would get combined map from all previous layers.

### CNN Hyperparameter

1. kernel size ( Height , width )

    The main convolution window which slides, since this is must requirement for the later weight matrix definition which is derived thought the size here. One by one convolutional window is also used but in hidden layer where it can work for dimensionality reduction along with introducing required non-linearity to the network for improving its overall learnings.

Output size formula
$$
Output = (\frac{Input + kernel}{stride})+1
$$

2. Padding

    Extra pixels added to the boundary of the kernel, making edges also come into the attention, keeping the output maps size equivalent to the input some nearby.

3. stride

    Pixels which to skip through, default is one, which also keeps the spatial information from the original image, with greater then one stride we can remove the need to pooling in layer layers

### Network Layers

1. Pooling Layer

    Withh pooling we can improve the models overall sensitivity with pixel position through out the network, with simple 3 variant of min, max, average pooling

2. Batch normalization

    removing sensitivity of layers overall stability through normalizing numerical values, improves learning overall with numerical flow being stable throughout.

    insensitive to weights initialization, acts as a form of regularization and helps netowork mentain numerical stability.

    *Implementation overview*
    with batch based $\mu$ and $\sigma$ calculation and applying normalization function with each of the element

    Keeping different shifting parameters $\alpha$ with $\beta$ for scale shifting of normalized weights, these both are learned into the training phase used with running statistics in inference mode.

3. Dropout Layer

    With a powerfull concept to improve existing connection work more robust at testing phase, we use dropout for masking activations to zero with uniform probability at training phase, making network more resilient and form stronger network connection with remaining connections.

    During training rest of non-de-activated nurons gets scalled weights to keep the training and testing conditions expections equal, leading for simplification and robust netowork architechture.


### Transfer learning 🧠

Training model from scratch needs huge dataset along with huge computational cost, due to which we can leverage the existing model for our needs.

### Using Existing model
+ Use pre-trained convolutional base
+ Remove original classifier head
+ Add new classifier head
+ Train only the new classifier head only

#### Basic ways to use pretrained weapon
+ Using them as the feature extractor > Ideal for small dataset
+ Fine Tuner : Initialize the network with pretrained weights

unfreeze few top layers or the whole network for the slight training with small learning rate would guide the model to be trained on the dataset.
Use with larger dataset availability.

## Transfer learning for segmentation

For segmentation (per pixel classifier), transfer learning is used to build the encoder of the encoder-decoder part of the network.

1. Architectures : U-Net and FCN works at the encoder part of the encoder-decoder part of the problem.

2. Pretrained Model : ResNet or VGG based 

3. Training the decoder : Takes the powerfull low-resolution features from the encoder and upsamples them to high resolution.
