<div align="center">    
 
# Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification     

[![Conference](http://img.shields.io/badge/ECCV-2018-4b44ce.svg)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)

</div>

This is the official GitHub page for the paper ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):


> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
In: *European Conference on Computer Vision (ECCV)*, Munich, Springer, 2018, 575-592.

## News

**12th December 2020 - [*PyTorch*](#Pytorch-Implementation) version:** 
We release a PyTorch implementation and provide weights of a pre-trained `base(M, f*)` model with underlying ResNet50 architecture.

**17th February 2020 - [*original_tf*](https://github.com/TIBHannover/GeoEstimation/tree/original_tf) branch:** 
Since our code is not compatible with TensorFlow 2 and relies on a Caffe model for scene classification, we have added a Dockerfile to simplify the installation. In addition, we have modified the inference script. It is now able to automatically generate the reported results in the paper for the testing datasets.


## Contents

- [Web Demo](#Demo) 
- [Reproduce Paper Results](#Reproduce-Paper-Results): A seperate branch (*original_tf*() where all information and pre-trained models are provided to reproduce the original results of the paper.
- [PyTorch Implementation](#PyTorch-Implementation): A re-implemented version of the `base(M, f*)` model in PyTorch including code for training, validation, testing, and inference.
- [Citation](#Citation)
- [License](#LICENSE)

## Demo
A graphical demonstration where you can compete against the deep learning approach presented in the publication can be found on: https://labs.tib.eu/geoestimation

This demo additionally supports uploading and analyzing your own images. A simplified offline version is located in this repository: https://tibhannover.github.io/GeoEstimation/



## Reproduce Paper Results 
We provide the original TensorFlow 1.14 implementation based on a ResNet101 to reproduce the results reported in the paper in the [*original_tf*](https://github.com/TIBHannover/GeoEstimation/tree/original_tf) branch.

## PyTorch Implementation
We have updated our code in PyTorch including scripts for training, inference and test. We have retrained the `base(M, f*)` model using a ResNet50 architecture and it achieved comparable results despite its lower complexity.

### Contents
- [Inference](#Inference)
- [Reproduce Results](#Reproduce-Results)
    - [Test on Already Trained Model](#Test-on-Already-Trained-Model)
    - [Training from Scratch](#Training-from-Scratch)
- [Geographical S2 Cell Partitioning](#Geographical-S2-Cell-Partitioning)
- [Installation and Requirements](#Requirements)

### Inference

To use the pre-trained model by default, first download the model checkpoint by running:
```
mkdir -p models/base_M
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/epoch.014-val_loss.18.4833.ckpt -O models/base_M/epoch=014-val_loss=18.4833.ckpt
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/hparams.yaml -O models/base_M/hparams.yaml
```

Inference with pre-trained model:
```bash
python -m classification.inference --image_dir resources/images/im2gps/
```

Available argparse parameter:
```
--checkpoint CHECKPOINT
    Checkpoint to already trained model (*.ckpt)
--hparams HPARAMS     
    Path to hparams file (*.yaml) generated during training
--image_dir IMAGE_DIR
    Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)
--gpu                 
    Use GPU for inference if CUDA is available, default to true
--batch_size BATCH_SIZE
--num_workers NUM_WORKERS
    Number of workers for image loading and pre-processing

```
Example output that is also stored in a *CSV* file:
```
img_id                                             p_key      pred_class  pred_lat   pred_lng 
Tokyo_00070_439717934_3d0fd200f1_180_97324495@N00  hierarchy  5367        41.4902    -81.7032
429881745_35a951f032_187_37718182@N00              hierarchy  8009        37.1770    -3.5877
104123223_7410c654ba_19_19355699@N00               hierarchy  7284        32.7337    -117.1520
```


### Reproduce Results

#### Test on Already Trained Model
The (list of) image files for testing can be found on the following links:
* Im2GPS: http://graphics.cs.cmu.edu/projects/im2gps/ (can be downloaded automatically)
* Im2GPS3k: https://github.com/lugiavn/revisiting-im2gps/

Download and extract the two testsets (Im2GPS, Im2GPS3k) in `resources/images/<dataset_name>` and run the evaluation script with the provided meta data, i.e., the ground-truth coordinate for each image.
When using the default paramters, make sure that the pre-trained model is available. 
```bash
# download im2gps testset
mkdir resources/images/im2gps
wget http://graphics.cs.cmu.edu/projects/im2gps/gps_query_imgs.zip -O resources/images/im2gps.zip
unzip resources/images/im2gps.zip -d resources/images/im2gps/

wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps_places365.csv -O resources/images/im2gps_places365.csv
wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv -O resources/images/im2gps3k_places365.csv
python -m classification.test
```

Available argparse paramters:
```
--checkpoint CHECKPOINT
    Checkpoint to already trained model (*.ckpt)
--hparams HPARAMS     
    Path to hparams file (*.yaml) generated during training
--image_dirs IMAGE_DIRS [IMAGE_DIRS ...]
    Whitespace separated list of image folders to evaluate
--meta_files META_FILES [META_FILES ...]
    Whitespace separated list of respective meta data (ground-truth GPS positions). Required columns: IMG_ID,LAT,LON
--gpu
    Use GPU for inference if CUDA is available, default to True
--precision PRECISION
    Full precision (32), half precision (16)
--batch_size BATCH_SIZE
--num_workers NUM_WORKERS
    Number of workers for image loading and pre-processing

```

Results on the Im2GPS and Im2GPS3k test sets: The reported accuracies (in percentage) is the fraction of images localized within the given radius (in km) using the GCD distance. Note, that we used the full MP-16 training dataset and all 25600 images for validation, thus the results will differ when not all images are available.

Im2GPS:
 Model   |    1 |   25 |   200 |   750 |   2500 | 
|:---------------|-----:|-----:|------:|------:|-------:
| base(M, c)     |  9.3 | 31.6 |  49.8 |  67.1 |   78.9 |
| base(M, m)     | 13.9 | 34.6 |  48.1 |  68.4 |   79.3 |
| base(M, f)     | 15.6 | 39.2 |  48.9 |  65.8 |   78.5 | 
| __base(M, f*)__    | 14.8 | 37.6 |  48.9 |  68.4 |   78.9 |
| __base(M, f*) (original)__     | 15.2 | 40.9 |  51.5 |  65.4 |   78.5 |

Im2GPS3k:

| Model   |    1 |   25 |   200 |   750 |   2500 | 
|:---------------|-----:|-----:|------:|------:|-------:
| base(M, c)     |  6.2 | 24.3 |  36.3 |  51.7 |   67.0 |
| base(M, m)     |  8.3 | 26.2 |  35.7 |  51.4 |   66.5 |
| base(M, f)     |  9.9 | 27.3 |  36.2 |  51.2 |   66.4 |
| __base(M, f*)__    | 10.1 | 28.0 |  36.9 |  51.1 |   67.0 |
| __base(M, f*) (original)__ | 9.7 | 27.0 | 35.6  | 49.2  | 66.0   |
| ISN(M, f*, S3) (original) | 10.5 | 28.0 | 36.6  | 49.7  | 66.0   |

#### Training from Scratch
We provide a complete training script which is written in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and report all hyper-paramters used for the provided model. Furthermore, a script is given to download and pre-process the images that are used for training and validiation.

1) Download training and validation images
    - We provide a script to download the images given a list of URLs
    - Due to no longer publicly available images, the size of the dataset might be smaller than the original.
    - We also store the images in chunks using [MessagePack](https://msgpack.org/) to speed-up the training process (similar to multiple TFRecord files)
2) Given multiple s2 partitionings (e.g. coarse, middle, fine from the paper), the respective classes are assigned to each image on both datasets.
3) Training and hyper-paramters: All hyper-paramters can be configured in `configs/baseM.yml` as well as paramters from PyTorch Lightning [`Trainer`](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api) class.


Necessary steps:
```bash
# download and preprocess images
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv -O resources/mp16_urls.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_urls.csv -O resources/yfcc25600_urls.csv 
python download_images.py --output resources/images/mp16 --url_csv resources/mp16_urls.csv --shuffle
python download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""

# assign cell(s) for each image using the original meta information
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv -O resources/mp16_places365.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv -O resources/yfcc25600_places365.csv
python partitioning/assign_classes.py
# remove images that were not downloaded 
python filter_by_downloaded_images.py

# train geo model from scratch
python -m classification.train_base --config config/baseM.yml
```

### Geographical S2 Cell Partitioning

The geographical cell labels are extracted using the [*S2 geometry library*](https://code.google.com/archive/p/s2-geometry-library/) and can be visualized on http://s2.sidewalklabs.com/regioncoverer/.
Create a partitioning using the following command for a given dataset (as *CSV* file) which contains an image id, latitude and longitude.
We provide the partitionings that are used in the paper [below](#Requirements).
```shell script
python partitioning/create_cells.py [-h] [-v] --dataset DATASET --output OUTPUT --img_min IMG_MIN --img_max IMG_MAX [--lvl_min LVL_MIN]
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
 
### Requirements
All requirements are listed in the `environment.yml`. We recomment to use [*conda*](https://docs.conda.io/en/latest/) to install all required packages in an individual environment.
```bash
# clone this repo
git clone https://github.com/TIBHannover/GeoEstimation.git && cd GeoEstimation
# install dependencies
conda env create -f environment.yml 
conda activate geoestimation-github-pytorch
# download pre-calculated parititonings
mkdir -p resources/s2_cells
wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_5000.csv -O resources/s2_cells/cells_50_5000.csv
wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_2000.csv -O resources/s2_cells/cells_50_2000.csv
wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_1000.csv -O resources/s2_cells/cells_50_1000.csv
```

## Citation
```BibTeX
@inproceedings{muller2018geolocation,
  author    = {Müller-Budack, Eric and Pustu-Iren, Kader and Ewerth, Ralph},
  title     = {Geolocation Estimation of Photos Using a Hierarchical Model and Scene
               Classification},
  booktitle = {Computer Vision - {ECCV} 2018 - 15th European Conference, Munich,
               Germany, September 8-14, 2018, Proceedings, Part {XII}},
  series    = {Lecture Notes in Computer Science},
  volume    = {11216},
  pages     = {575--592},
  publisher = {Springer},
  year      = {2018},
  url       = {https://doi.org/10.1007/978-3-030-01258-8\_35},
  doi       = {10.1007/978-3-030-01258-8\_35},
}
```

## Licence
This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.
