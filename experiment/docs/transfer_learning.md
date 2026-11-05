# Transfer learning

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
