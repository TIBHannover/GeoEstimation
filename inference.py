import argparse
import glob
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses unnecessarily excessive console output
import tensorflow as tf

# own imports
import scene_classification
import geo_estimation

# VARIABLES
cur_path = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', nargs='+', type=str, required=True, help='path to image file(s)')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='ISN',
        choices=['ISN', 'base_L', 'base_M'],
        help='Choose from [ISN, base_L, base_M]')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('-s', '--show_cam', action='store_true', help='show class activation map')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # check if gpu is available
    if not tf.test.is_gpu_available():
        print('No GPU available. Using CPU instead ... ')
        args.cpu = True

    # get images located in input
    img_list = []
    for input_path in args.input:
        for f in glob.iglob(input_path + '/**/*.jpg', recursive=True):
            img_list.append(f)

    print('{} images found in {}'.format(len(img_list), args.input))

    # init models
    if args.model == 'ISN':
        # init model for scene_classification
        sc = scene_classification.SceneClassifier(use_cpu=args.cpu)

        # init models for geolocation estimation
        # init ISN for concept 'indoor'
        ge_indoor = geo_estimation.GeoEstimator(
            os.path.join(cur_path, 'models', 'ISN_M_indoor', 'model.ckpt'), scope='indoor', use_cpu=args.cpu)
        # init ISN for concept 'natural'
        ge_natural = geo_estimation.GeoEstimator(
            os.path.join(cur_path, 'models', 'ISN_M_natural', 'model.ckpt'), scope='natural', use_cpu=args.cpu)
        # init ISN for concept 'urban'
        ge_urban = geo_estimation.GeoEstimator(
            os.path.join(cur_path, 'models', 'ISN_M_urban', 'model.ckpt'), scope='urban', use_cpu=args.cpu)
    elif args.model == 'base_L':
        ge_base = geo_estimation.GeoEstimator(
            os.path.join(cur_path, 'models', 'base_L_m', 'model.ckpt'), scope='base_L_m', use_cpu=args.cpu)
    elif args.model == 'base_M':
        ge_base = geo_estimation.GeoEstimator(
            os.path.join(cur_path, 'models', 'base_M', 'model.ckpt'), scope='base_M', use_cpu=args.cpu)

    for img_file in img_list:
        print('Processing: {}'.format(img_file))

        # predict scene label
        if args.model == 'ISN':
            # get scene label
            scene_probabilities = sc.get_scene_probabilities(img_file)
            scene_label = sc.get_scene_label(scene_probabilities)
        else:
            scene_label = -1

        # predict geolocation depending on model and scenery
        if scene_label == -1:
            result = ge_base.get_prediction(img_file, show_cam=args.show_cam)
        if scene_label == 0:
            result = ge_indoor.get_prediction(img_file, show_cam=args.show_cam)
        if scene_label == 1:
            result = ge_natural.get_prediction(img_file, show_cam=args.show_cam)
        if scene_label == 2:
            result = ge_urban.get_prediction(img_file, show_cam=args.show_cam)

        # save results

    return 0


if __name__ == '__main__':
    sys.exit(main())
