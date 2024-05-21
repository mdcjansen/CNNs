# Preprocessing data
This folder contains several scripts that aid in processing image data, before the CNN scripts are used to train the 
models.

## Table of Contents
* [Auto exclusion](#auto-exclusion)
* [Colour augmentation](#colour-augmentation)
* [File sorting](#file-sorting)
* [Multiprocess colour augmentation](#multiprocess-colour-augmentation)
* [Kfold crossvalidation](#kfold-cross-validation)
* [Marchenko normalization](#Marchenko-normalization)
* [Mask filtering](#mask-filtering)
* [Oversampling](#oversampling)
* [Parameter loading](#parameter-loading)
* [Patch extraction](#patch-extraction)
* [Default parameters csv](#default-parameters)
* [binary classification xlsx](#binary-classification-xlsx)

## Auto exclusion


## Colour augmentation
The first version of augmenting input images. These images are augmented in batches of ten images at any given time. The
brightness, contrast, saturation, and hue are changed by a set value, which can be changed by the user in line 39. This 
script was developed to allow for direct adaptation into the CNN codes, once images could successfully be colour 
augmented. The current version is no longer present in the CNN scripts. However, this script will function on its own. 
All input images present within the input folder, will be colour augmented and saved as a separate image within the 
output folder.

## File sorting


## Folder sorting


## Multiprocess colour augmentation
This multiprocessing version of colour augmentation is incorporated into the CNN scripts. Several changes have been made
in comparison to the [colour augmentation](#colour-augmentation) script.

Firstly, a range of values now has to be provided instead of a fixed value for augmentation. Providing a broader range 
of augmented images. Secondly, specific image formats can be provided for processing. By default, only '.jpg' files are 
processed. Lastly, current time will be given at each print. Allowing for simplistic time keeping, when each step has 
been performed.

## Kfold cross validation
Based on user input, multiple kfold cross validation folds and a test set are produced. By default, a test set is 
created that contains 30% of the input data. Additionally, four kfold cross validated folds are produced.

The script requires and input folder, containing the images to create folds for, an output folder that will contain the 
folds as well as the test set, and the [binary label file](#binary-classification-xlsx). This file is used to maintain 
the same class distribution in the folds, as is present in the input folder.

## Marchenko normalization
Marchenko normalization is applied on the input images. The normalization is based off a reference image, which should 
not be part of the dataset that will be used to train the CNNs. To perform the normalization, an input folder, output 
folder, and reference image must be provided by the user. The input images are not modified during normalization, 
instead normalized images are saved as new '.jpg' images.

## Mask filtering
This script filters the input images, based on their accompanying mask size. Images that have a masked size percentage 
lower than the input value, are moved to the output folder. By default, images with less than 75% mask coverage will be 
filtered out. The script requires an input, output, and a value between 0.0 and 1.0 for the mask size threshold 
(default= 0.75).

## Oversampling
This script will perform oversampling on a set of images to remove class imbalance from two classes. Oversampling is 
performed by randomly selecting input images and created copies that are horizontally, vertically, or hybrid flipped. 
Where images with a hybrid flip are both horizontally and vertically flipped. This script requires an input folder which 
contains the images to be oversampled. As well as an output folder and the binary classification file, explained in
detail [here]().

## Parameter loading
This debug script was used to test the functionality of loading all input parameters for the CNNs through a 
[csv file](). This script has been adapted into the CNN scripts to handle loading the parameters. 
Currently, this script will load the parameters into the correct data format that is required by the CNN scrips. 
all parameters can be changed, used, or printed as needed by the user.

## Patch extraction
This groovy script is used to extract the images used by the CNN from a larger image. The user has to provide at which 
magnification the extraction is performed, as well as the output folder, image format, and image dimensions. Only images
that have a accompanying label image with a partially or completely coloured background (non-white) will be extracted 
and saved in the output folder.
