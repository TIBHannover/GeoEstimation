import argparse
import caffe
import csv
import json
import numpy as np
import os
import s2sphere as s2
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses unnecessarily excessive console output
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import cnn_architectures


class GeoEstimator():

    def __init__(self, model_file, cnn_input_size=224, scope=None, use_cpu=False):
        print('Initialize {} geolocation model.'.format(scope))

        self._cnn_input_size = cnn_input_size

        # load model config
        with open(os.path.join(os.path.dirname(model_file), 'cfg.json')) as cfg_file:
            cfg = json.load(cfg_file)

        # get partitioning
        print('\tGet geographical partitioning(s) ... ')
        partitioning_files = []
        for partitioning in cfg['partitionings']:
            partitioning_files.append(os.path.join(os.path.dirname(__file__), 'geo-cells', partitioning))

        self._num_partitionings = len(partitioning_files)

        # red geo partitioning
        classes_geo, hexids2classes, class2hexid, cell_centers = self._read_partitioning(partitioning_files)

        self._classes_geo = classes_geo
        self._cell_centers = cell_centers

        # get geographical hierarchy
        self._cell_hierarchy = self._get_geographical_hierarchy(classes_geo, hexids2classes, class2hexid, cell_centers)

        # build cnn
        self._image_ph = tf.placeholder(shape=[3, self._cnn_input_size, self._cnn_input_size, 3], dtype=tf.float32)

        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        print('\tRestore model from: {}'.format(model_file))

        if scope is not None:
            with tf.variable_scope(scope) as scope:
                self.scope = scope
        else:
            self.scope = tf.get_variable_scope()

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.variable_scope(self.scope):
            with tf.device(device):
                net, _ = cnn_architectures.create_model(
                    cfg['architecture'], self._image_ph, is_training=False, num_classes=None, reuse=None)

                with tf.variable_scope('classifier_geo', reuse=None):
                    self.logits = slim.conv2d(
                        net, np.sum(classes_geo), [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    self.logits = tf.squeeze(self.logits)

        var_list = {x.name.replace(self.scope.name + '/', '')[:-2]: x for x in tf.global_variables(self.scope.name)}
        saver = tf.train.Saver(var_list=var_list)

        saver.restore(self.sess, model_file)

    def get_prediction(self, img_path):
        # read image
        img_path_ph = tf.placeholder(shape=[], dtype=tf.string)
        img_content = tf.read_file(img_path)

        # decode image
        img = tf.image.decode_jpeg(img_content, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        # normalize image to -1 .. 1
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        # crop image into three evenly sampled crops
        img_crops = self._crop_image(img)

        # apply image transformations
        img_crops_v = self.sess.run(img_crops, feed_dict={img_path_ph: img_path})

        # feed forward batch of images in cnn and extract result
        x = self.sess.run(self.logits, feed_dict={self._image_ph: img_crops_v})

        # softmax to get class probabilities with sum 1
        x[0, :] = self._softmax(x[0, :])
        x[1, :] = self._softmax(x[1, :])
        x[2, :] = self._softmax(x[2, :])

        # fuse results of image crops using the maximum
        logits = np.max(x, axis=0)

        # assign logits to respective partitionings and get prediction (class with highest probability)
        partitioning_logits = []
        partitioning_pred = []
        for p in range(self._num_partitionings):
            p_logits = logits[np.sum(self._classes_geo[0:p + 1]):np.sum(self._classes_geo[0:p + 2])]
            p_pred = p_logits.argsort()
            partitioning_logits.append(p_logits)
            partitioning_pred.append(p_pred[-1])

        # get hierarchical multipartitioning results
        hierarchical_logits = partitioning_logits[-1]  # get logits from finest partitioning
        if self._num_partitionings > 1:
            for c in range(self._classes_geo[-1]):  # num_logits of finest partitioning
                for p in range(self._num_partitionings - 1):
                    hierarchical_logits[c] *= partitioning_logits[p][self._cell_hierarchy[c][p]]

            pred = hierarchical_logits.argsort()
            partitioning_pred.append(pred[-1])

        prediction = partitioning_pred[-1]

        # get gps coordinate from class
        lat, lng = self._cell_centers[self._num_partitionings - 1][prediction]
        print('Predicted coordinate (lat, lng): ({}, {})'.format(lat, lng))
        return lat, lng

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _crop_image(self, img):
        height = tf.to_float(tf.shape(img)[0])
        width = tf.to_float(tf.shape(img)[1])

        # get minimum and maximum coordinate
        max_side_len = tf.maximum(width, height)
        min_side_len = tf.minimum(width, height)
        is_w, is_h = tf.cond(tf.less(width, height), lambda: (0, 1), lambda: (1, 0))

        # resize image
        ratio = self._cnn_input_size / min_side_len
        offset = (tf.to_int32(max_side_len * ratio + 0.5) - self._cnn_input_size) // 2
        img = tf.image.resize_images(img, size=[tf.to_int32(height * ratio + 0.5), tf.to_int32(width * ratio + 0.5)])

        # get crops according to image orientation
        img_array = []
        for i in range(3):
            img_crop = tf.image.crop_to_bounding_box(img, i * is_h * offset, i * is_w * offset, self._cnn_input_size,
                                                     self._cnn_input_size)
            img_crop = tf.expand_dims(img_crop, 0)
            img_array.append(img_crop)

        return tf.concat(img_array, axis=0)

    def _read_partitioning(self, partitioning_files):
        # define vars
        cell_centers = []  # list of cell centers for each class
        classes_geo = []  # list of number of geo_classes for each partitioning
        hexids2classes = []  # dictionary to convert a hexid into a class label
        class2hexid = []  # hexid for each class
        classes_geo.append(0)  # add zero to classes vector for simplification in further processing steps

        # get cell partitionings
        for partitioning in partitioning_files:
            partitioning_hexids2classes = {}
            partitioning_classes_geo = 0
            partitioning_class2hexid = []
            partitioning_cell_centers = []

            with open(partitioning, 'r') as cell_file:
                cell_reader = csv.reader(cell_file, delimiter=',')
                for line in cell_reader:
                    if len(line) > 1 and line[0] not in ['num_images', 'min_concept_probability', 'class_label']:
                        partitioning_hexids2classes[line[1]] = int(line[0])
                        partitioning_class2hexid.append(line[1])
                        partitioning_cell_centers.append([float(line[3]), float(line[4])])
                        partitioning_classes_geo += 1

                hexids2classes.append(partitioning_hexids2classes)
                class2hexid.append(partitioning_class2hexid)
                cell_centers.append(partitioning_cell_centers)
                classes_geo.append(partitioning_classes_geo)

        return classes_geo, hexids2classes, class2hexid, cell_centers

    # generate hierarchical list of respective higher-order geo cells for multipartitioning [|c| x |p|]
    def _get_geographical_hierarchy(self, classes_geo, hexids2classes, class2hexid, cell_centers):
        cell_hierarchy = []

        if self._num_partitionings > 1:
            # loop through finest partitioning
            for c in range(classes_geo[-1]):
                cell_bin = self._hextobin(class2hexid[-1][c])
                level = int(len(cell_bin[3:-1]) / 2)
                parents = []

                # get parent cells
                for l in reversed(range(2, level + 1)):
                    hexid_parent = self._create_cell(cell_centers[-1][c][0], cell_centers[-1][c][1], l)
                    for p in reversed(range(self._num_partitionings - 1)):
                        if hexid_parent in hexids2classes[p]:
                            parents.append(hexids2classes[p][hexid_parent])

                    if len(parents) == self._num_partitionings - 1:
                        break

                cell_hierarchy.append(parents[::-1])

        return cell_hierarchy

    def _hextobin(self, hexval):
        thelen = len(hexval) * 4
        binval = bin(int(hexval, 16))[2:]
        while ((len(binval)) < thelen):
            binval = '0' + binval

        binval = binval.rstrip('0')
        return binval

    def _hexid2latlng(self, hexid):
        # convert hexid to latlng of cellcenter
        cellid = s2.CellId().from_token(hexid)
        cell = s2.Cell(cellid)
        point = cell.get_center()
        latlng = s2.LatLng(0, 0).from_point(point).__repr__()
        _, latlng = latlng.split(' ', 1)
        lat, lng = latlng.split(',', 1)
        lat = float(lat)
        lng = float(lng)
        return lat, lng

    def _latlng2class(self, lat, lng, hexids2classes):
        for l in range(2, 18):  # NOTE: upper boundary necessary
            hexid = create_cell(lat, lng, l)
            if hexid in hexids2classes:
                return hexids2classes[hexid]

    def _create_cell(self, lat, lng, level):
        p1 = s2.LatLng.from_degrees(lat, lng)
        cell = s2.Cell.from_lat_lng(p1)
        cell_parent = cell.id().parent(level)
        hexid = cell_parent.to_token()
        return hexid


'''
########################################################################################################################
# MAIN TO TEST CLASS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--image', type=str, required=True, help='path to image file')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model file')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # check if gpu is available
    if not tf.test.is_gpu_available():
        print('No GPU available. Using CPU instead ... ')
        args.cpu = True

    # init scene classifier
    ge = GeoEstimator(args.model, scope='geo', use_cpu=args.cpu)

    # predict scene label
    pred = ge.get_prediction(args.image)

    return 0


if __name__ == '__main__':
    sys.exit(main())
