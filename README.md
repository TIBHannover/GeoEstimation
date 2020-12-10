# Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification
This is the official GitHub page for the paper ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):

> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
In: *European Conference on Computer Vision (ECCV)*, Munich, Springer, 2018, 575-592.

## News

**12th December 2020:**
Restructured entire project. In addition, we release a PyTorch implementation and provide weights of a pre-trained base(M, f*) model.


## Demo

A graphical demonstration where you can compete against the deep learning approach presented in the publication can be found on: https://tibhannover.github.io/GeoEstimation/

We also created an extended web-tool that additionally supports uploading and analyzing your own images: https://labs.tib.eu/geoestimation

## Content

This repository contains:
- Branch `original_tf` where all information and pre-trained models are provided to reproduce the original results of the paper.
- Branch `pytorch` where we provide a PyTorch implementation of the baseM model. This includes training, validation, testing, and inference.
- This branch holds the demo and a script to create your own s2 partitionings.

## Geographical S2 Cell Partitioning

The geographical cell labels are extracted using the *S2 geometry library*:
https://code.google.com/archive/p/s2-geometry-library/

The geographical cells can be visualized on:
http://s2.sidewalklabs.com/regioncoverer/

Create a partitioning using the following command for a given dataset (as CSV) which contains an image id, latitude and longitude:
```shell script
python create_cells.py [-h] [-v] --dataset DATASET --output OUTPUT --img_min IMG_MIN --img_max IMG_MAX [--lvl_min LVL_MIN]
                       [--lvl_max LVL_MAX]
# Optional arguments:
#   -h, --help         show this help message and exit
#   -v, --verbose      verbose output
#   --dataset DATASET  Path to dataset csv file
#   --output OUTPUT    Path to output directory
#   --img_min IMG_MIN  Minimum number of images per geographical cell
#   --img_max IMG_MAX  Maximum number of images per geographical cell
#   --lvl_min LVL_MIN  Minimum partitioning level (default = 2)
#   --lvl_max LVL_MAX  Maximum partitioning level (default = 30)
#   --column_img_path  CSV input column name for image id / path
#   --column_lat       CSV input column name latitude
#   --column_lng       CSV input column name longitude
```

## LICENSE

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
