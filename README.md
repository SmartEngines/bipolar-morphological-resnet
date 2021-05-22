# bipolar-morphological-resnet

This is TensorFlow code for experiments on MNIST and CIFAR-10 bipolar morphological ResNet-like neural network (https://arxiv.org/abs/2009.07190).

mnist_morph.py - the main file for MNIST experiment

cifar10_morph.py - the main file for CIFAR-10 experiment

BipolarMorphologicalConv2D.py - the file with bipolar morphological convolutional layer implementation

resnet.py - the file with BM-ResNet network implementation

utils.py - the file with helper functions

The implementation of ResNet network is based on the code from Keras https://keras.io/examples/cifar10_resnet/

In pretrained_models folder we share:

mnist_ResNet20v2_model.22bm.h5 - ResNet for MNIST with 22 BM convolutional layers

cifar10_ResNet20v2_model.22bm.h5 - ResNet for CIFAR10 with 22 BM convolutional layers

cifar10_ResNet20v2_model.16bm.h5 - ResNet for CIFAR10 with 16 BM convolutional layers
