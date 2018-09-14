# Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification
This is the official GitHub page for the paper ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):

> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
Forthcoming: *European Conference on Computer Vision (ECCV).* Munich, 2018.

# Demo

A graphical demonstration where you can compete against the deep learning approach presented in the publication can be found on: https://tibhannover.github.io/GeoEstimation/

# Content

This repository contains:
* Meta information for the
[MP-16](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv)
training dataset as well as the [Im2GPS](meta/im2gps_places365.csv) and
[Im2GPS3k](meta/im2gps3k_places365.csv) test datasets:
    - Relative image path containing the Flickr-ID
    - Flickr Author-ID
    - Ground-truth latitude
    - Ground-truth longitude
    - Predicted scene label in *S_3*
    - Predicted scene label in *S_16*
    - Predicted scene label in *S_365*
    - Probability for *S_3* concept *indoor*
    - Probability for *S_3* concept *natural*
    - Probability for *S_3* concept *urban*
* List of geographical cells for all partitionings
([coarse](geo-cells/cells_50_5000.csv),
[middle](geo-cells/cells_50_2000.csv),
[fine](geo-cells/cells_50_1000.csv))
    - Class label
    - Hex-ID according to the *S2 geometry library*
    - Number of images in the geo-cell
    - Mean latitude of all images in the geo-cell
    - Mean longitude of all images in the geo-cell
* Results for the reported approaches on [Im2GPS](results/im2gps) and [Im2GPS3k](results/im2gps3k) <approach_parameters.csv>:
    - Relative image path containing the Flickr-ID
    - Ground-truth latitude
    - Predicted latitude
    - Ground-truth longitude
    - Predicted longitude
    - Great-circle distance (GCD)

# Images

The (list of) image files for training and testing can be found on the following links:
* MP-16: http://multimedia-commons.s3-website-us-west-2.amazonaws.com/
* Im2GPS: http://graphics.cs.cmu.edu/projects/im2gps/
* Im2GPS-3k: https://github.com/lugiavn/revisiting-im2gps

# Geographical Cell Partitioning

The geographical cell labels are extracted using the *S2 geometry library*:
https://code.google.com/archive/p/s2-geometry-library/

The python implementation *s2sphere* can be found on:
http://s2sphere.readthedocs.io/en/

The geographical cells can be visualized on:
http://s2.sidewalklabs.com/regioncoverer/

# Scene Classification

The scene labels and probabilities are extracted using the *Places365 ResNet 152 model* from:
https://github.com/CSAILVision/places365

In order to generate the labels for the superordinate scene categories the *Places365 hierarchy* is used:
http://places2.csail.mit.edu/download.html

# Geolocation Models

All models were trained using TensorFlow

* Baseline approach for middle partitioning: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_L_m.tar.gz)
* Multi-partitioning baseline approach: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_M.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *indoor*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_indoor.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *natural*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_natural.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *urban*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_urban.tar.gz)

We are currently working on a deploy source code.

# Requirements
Please make sure to have the following python3 libraries installed:
* caffe (pycaffe)
* csv
* numpy
* s2sphere
* tensorflow


# Installation
1. Clone this repository:
```
git clone git@github.com:TIBHannover/GeoEstimation.git
```
2. Either use the provided downloader using ```python downloader.py``` to get all necessary files or follow these instructions:
    * Download the Places365 ResNet 152 model for scene classification as well as the hierarchy file ([Links](#scene-classification)) and save all files in a new folder called */resources*
    * Download and extract the TensorFlow model files ([Links](#geolocation-models)) for geolocation and save them in a new folder called */models*.
3. Run the inference script by executing the following command with an image of your choice:
```
python inference.py -i <PATH/TO/IMG/FILE>
```
You can choose one of the following models for geolocatization: *Model=[ISN, base_M, base_L_m]*. *ISN* are the standard model.
```
python inference.py -i <PATH/TO/IMG/FILE> -m <MODEL>
```
If you want to run the code on the cpu, please execute the following command:
```
python inference.py -i <PATH/TO/IMG/FILE> -m <MODEL> -c
```

# LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
