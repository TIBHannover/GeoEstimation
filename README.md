# Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification
This is the official GitHub page for the paper ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):

> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
In: *European Conference on Computer Vision (ECCV)*, Munich, Springer, 2018, 575-592.

## Demo

A graphical demonstration where you can compete against the deep learning approach presented in the publication can be found on: https://tibhannover.github.io/GeoEstimation/

We also created an extended web-tool that additionally supports uploading and analyzing your own images: https://labs.tib.eu/geoestimation

## Content

This repository contains:
- Meta information for the
[MP-16](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv)
training dataset as well as the [Im2GPS](meta/im2gps_places365.csv) and
[Im2GPS3k](meta/im2gps3k_places365.csv) test datasets:
- List of geographical cells for all partitionings:
[coarse](geo-cells/cells_50_5000.csv),
[middle](geo-cells/cells_50_2000.csv),
[fine](geo-cells/cells_50_1000.csv)
- Results for the reported approaches on [Im2GPS](results/im2gps) and [Im2GPS3k](results/im2gps3k) <approach_parameters.csv>:
- A python script to download all necessary resources to run the scripts `downloader.py`
- Inference script to reproduce the paper results `inference.py`

## Training and Testing Images

The (list of) image files for training and testing can be found on the following links:
* MP-16: http://multimedia-commons.s3-website-us-west-2.amazonaws.com/
* Im2GPS: http://graphics.cs.cmu.edu/projects/im2gps/
* Im2GPS-3k: https://github.com/lugiavn/revisiting-im2gps/

## Geographical Cell Partitioning

The geographical cell labels are extracted using the *S2 geometry library*:
https://code.google.com/archive/p/s2-geometry-library/

The python implementation *s2sphere* can be found on:
http://s2sphere.readthedocs.io/en/

The geographical cells can be visualized on:
http://s2.sidewalklabs.com/regioncoverer/

## Scene Classification

The scene labels and probabilities are extracted using the *Places365 ResNet 152 model* from:
https://github.com/CSAILVision/places365

In order to generate the labels for the superordinate scene categories the *Places365 hierarchy* is used:
http://places2.csail.mit.edu/download.html

## Geolocation Models

All models were trained using TensorFlow (1.14)

* Baseline approach for middle partitioning: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_L_m.tar.gz)
* Multi-partitioning baseline approach: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/base_M.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *indoor*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_indoor.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *natural*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_natural.tar.gz)
* Multi-partitioning Individual Scenery Network for *S_3* concept *urban*: [Link](https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/ISN_M_urban.tar.gz)

## Installation

1. Either use the provided script using ```python downloader.py``` to get all necessary files or follow these instructions:
    * Download the ResNet 152 model for scene classification trained on Places365 as well as the hierarchy file ([Links](#scene-classification)) and save all files in a new folder called */resources*
    * Download and extract the TensorFlow model files ([Links](#geolocation-models)) for geolocation and save them in a new folder called */models*.
2. We provide a docker container to run the code:
    ```shell script
    docker build <PROJECT_FOLDER> -t <DOCKER_NAME>
    docker run \
        --volume <PATH/TO/PROJECT/FOLDER>:/src \
        --volume <PATH/TO/IMAGE/FILES>:/img \
        -u $(id -u):$(id -g) -it <DOCKER_NAME> bash
    ```


## Inference

Run the inference script by executing the following command with an image of your choice:
```shell script
python inference.py -i <PATH/TO/IMG/FILE>
```
or for a list of images with e.g.:
```shell script
python inference.py -i <PATH/TO/IMG/FILES/*.JPG>
```
You can choose one of the following models for geolocalization: *Model=[base_L, base_M, ISN]*. *ISNs* are the standard models.
```shell script
python inference.py -i <PATH/TO/IMG/FILES/*.JPG> -m <MODEL>
```
In order to reproduce our paper results, [download the images](#training-and-testing-images) and provide the meta information file for [Im2GPS](meta/im2gps_places365.csv) or [Im2GPS3k](meta/im2gps3k_places365.csv). Note, that the image filename has to correspond to the `IMG_ID` in the meta information and run the following command:
```shell script
python inference.py -i <PATH/TO/IMG/FILES/*.JPG> -m <MODEL> -l <PATH/TO/META/INFORMATION>
```
The visualization of class activation maps can be enabled using ```-s```:
```shell script
python inference.py -i <PATH/TO/IMG/FILES/*.JPG> -s
```
If you want to run the code on the cpu, please execute the script using the flag ```-c```:
```shell script
python inference.py -i <PATH/TO/IMG/FILES/*.JPG> -m <MODEL> -c
```

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
