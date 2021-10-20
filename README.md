# This repository contains deep learning based computer vision example experiments. 

## Requirements
Python (>=3.6), PyTorch (>=1.9), transformers (4.11.3).

## Datasets
The used datasets are MNIST, SVHN and STL10.

## Implemented Networks
The following different types of networks are implement for multiclass classification task. 
1. Base model (classfication model) that uses STN (Spatial Transformer Network).
2. Base model that uses STN along with CoordConv layers (by replacing all the nn.Conv2d layers)
3. Base model that uses STN along with deformable ConvNets (v2) (network contains both nn.Conv2d and deformable ConvNets)
4. ViT Network (Visual Transformer Network) model + STN 


## Model Performance Evaluation
Qualitative evaluation of the STN network's performance is performed by plotting original and transformed images. 

The classifcation performance is evaluated with ROC and Confusion Matrix plot. 
ROC is suitable when classes are balanced, which is the case in this experiment.
Confusion matrix depicts the how the classes are correctly and incorectly classified.
In the case of MNIST, confusion matrix can tell which digits are difficult to classify because of their similar structural appearance.

The models store the images in the directory $ROOT_DIR$ + "figures/"

## Setup
Run the ``` main.py``` file. 
It will install all required datasets to $ROOT_DIR$ + "data/" location. 
