import argparse
import csv
import numpy as np
import os
import sys

os.environ['GLOG_minloglevel'] = '3'  # Suppresses unnecessarily excessive console output
import caffe
import tensorflow as tf

cur_dir = os.path.abspath(os.path.dirname(__file__))


class SceneClassifier():

    def __init__(self,
                 prototxt_file=os.path.join(cur_dir, 'resources', 'deploy_resnet152_places365.prototxt'),
                 caffemodel_file=os.path.join(cur_dir, 'resources', 'resnet152_places365.caffemodel'),
                 scene_hierarchy_file=os.path.join(cur_dir, 'resources', 'scene_hierarchy_places365.csv'),
                 use_cpu=True):

        # read scene_hierarchy file to get lvl1 meta information
        print('Load scene hierarchy ... ')
        hierarchy_places3 = []
        with open(scene_hierarchy_file, 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content)  # skip explanation line
            next(content)  # skip explanation line
            for line in content:
                hierarchy_places3.append(line[1:4])

        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=np.float)

        # normalize label if it belongs to multiple categories
        self.hierarchy_places3 = hierarchy_places3 / np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)

        print('Restore scene classification model from: {}'.format(caffemodel_file))

        # initialize network
        if use_cpu:
            caffe.set_mode_cpu()  # CPU
        else:
            caffe.set_mode_gpu()  # GPU
            caffe.set_device(0)

        self.net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)

        # steps for image preprocessing
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer.set_raw_scale('data', 255.0)

        # define network input dimensions
        self.net.blobs['data'].reshape(1, 3, 224, 224)

    def get_scene_probabilities(self, img_path):
        # load and preprocess image
        img = caffe.io.load_image(img_path)
        img = self.transformer.preprocess('data', img)

        # feed image data into cnn
        self.net.blobs['data'].data[...] = [img]
        scene_probs = self.net.forward()['prob']

        # get the probabilites of the lvl 1 categories
        places3_prob = np.matmul(scene_probs, self.hierarchy_places3)[0]

        return places3_prob

    def get_scene_label(self, scene_prob):
        scene_label_int = np.argmax(scene_prob, axis=0)
        if scene_label_int == 0:
            return 'indoor'
        elif scene_label_int == 1:
            return 'natural'
        else:  # scene_label_int == 2:
            return 'urban'


'''
########################################################################################################################
# MAIN TO TEST CLASS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--image', type=str, required=True, help='path to image file')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # init scene classifier
    if not tf.test.is_gpu_available():
        print('No GPU available. Using CPU instead ... ')
        args.cpu = True

    sc = SceneClassifier(use_cpu=args.cpu)

    # predict scene label
    places3_prob = sc.get_scene_probabilities(args.image)
    places3_label = sc.get_scene_label(places3_prob)

    return 0


if __name__ == '__main__':
    sys.exit(main())
