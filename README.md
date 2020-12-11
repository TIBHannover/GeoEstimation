# Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification
This is the official GitHub page for the paper ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):

> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
In: *European Conference on Computer Vision (ECCV)*, Munich, Springer, 2018, 575-592.

## News

**12th December 2020 - [`pytorch`](https://github.com/TIBHannover/GeoEstimation/tree/pytorch) branch:** 
We release a PyTorch implementation and provide weights of a pre-trained `base(M, f*)` model.

**17th February 2020 - [`original_tf`](https://github.com/TIBHannover/GeoEstimation/tree/original_tf) branch:** 
Since our code is not compatible with TensorFlow 2 and relies on a Caffe model for scene classification, we have added a Dockerfile to simplify the installation. In addition, we have modified the inference script. It is now able to automatically generate the reported results in the paper for the testing datasets.

## Content
This repository contains:

- Branch [`original_tf`](#Reproduce-Paper-Results) where all information and pre-trained models are provided to reproduce the original results of the paper.
- Branch [`pytorch`](#PyTorch-Implementation) where we provide a PyTorch implementation reproducing the `base(M, f*)` model. This includes training, validation, testing, and inference.
- [Web demo](#Demo) 

## Reproduce Paper Results 
We provide the original TensorFlow 1.14 implementation based on a ResNet101 to reproduce the results reported in the paper in the [`original_tf`](https://github.com/TIBHannover/GeoEstimation/tree/original_tf) branch.

## PyTorch Implementation
We have updated our code in PyTorch including scripts for training, inference and test. We have retrained the `base(M, f*)` model using a ResNet50 architecture and it achieved comparable results despite its lower complexity. The code is provided in the [`pytorch`](https://github.com/TIBHannover/GeoEstimation/tree/pytorch) branch.


## Demo

A graphical demonstration where you can compete against the deep learning approach presented in the publication can be found on: https://tibhannover.github.io/GeoEstimation/

We also created an extended web-tool that additionally supports uploading and analyzing your own images: https://labs.tib.eu/geoestimation

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
