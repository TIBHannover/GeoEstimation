import argparse
import csv
import json
import numpy as np
import os
import re
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

        self._image_path_placeholder = tf.placeholder(tf.string, shape=())
        image_content = tf.read_file(self._image_path_placeholder)
        self._image_crops, self._bboxes = self._img_preprocessing(image_content)

        # load model config
        with open(os.path.join(os.path.dirname(model_file), 'cfg.json')) as cfg_file:
            cfg = json.load(cfg_file)

        # get partitioning
        print('\tGet geographical partitioning(s) ... ')
        partitioning_files = []
        partitionings = []
        for partitioning in cfg['partitionings']:
            partitioning_files.append(os.path.join(os.path.dirname(__file__), 'geo-cells', partitioning))
            partitionings.append(partitioning)
        if len(partitionings) > 1:
            partitionings.append('hierarchy')

        self._num_partitionings = len(partitioning_files)  # without hierarchy

        # red geo partitioning
        classes_geo, hexids2classes, class2hexid, cell_centers = self._read_partitioning(partitioning_files)

        self._classes_geo = classes_geo
        self._cell_centers = cell_centers

        # get geographical hierarchy
        self._cell_hierarchy = self._get_geographical_hierarchy(classes_geo, hexids2classes, class2hexid, cell_centers)

        # build cnn
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        print('\tRestore model from: {}'.format(model_file))

        if scope is not None:
            with tf.variable_scope(scope) as scope:
                self._scope = scope
        else:
            self._scope = tf.get_variable_scope()

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.variable_scope(self._scope):
            with tf.device(device):
                net, _ = cnn_architectures.create_model(
                    cfg['architecture'], self._image_crops, is_training=False, num_classes=None, reuse=None)

                with tf.variable_scope('classifier_geo', reuse=None):
                    self.logits = slim.conv2d(
                        net, np.sum(classes_geo), [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    self.logits = tf.squeeze(self.logits)

        var_list = {
            re.sub('^' + self._scope.name + '/', '', x.name)[:-2]: x for x in tf.global_variables(self._scope.name)
        }

        # restore weights
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self._sess, model_file)

        # get activations from last conv layer and output weights in order to calculate class activation maps
        self._activations = tf.get_default_graph().get_tensor_by_name(self._scope.name +
                                                                      '_1/resnet_v2_101/activations:0')
        activation_weights = tf.get_default_graph().get_tensor_by_name(self._scope.name +
                                                                       '/classifier_geo/logits/weights:0')

        activation_weights_v = self._sess.run(activation_weights)
        p_activation_weights = []
        for p in range(self._num_partitionings):
            p_activation_weights.append(
                activation_weights_v[:, :, :,
                                     np.sum(self._classes_geo[0:p + 1]):np.sum(self._classes_geo[0:p + 2])])

        self.network_dict = {
            'activation_weights': p_activation_weights,
            'partitionings': partitionings,
            'classes_geo': self._classes_geo,
            'cell_centers': self._cell_centers,
            'scope': self._scope.name
        }

    def _img_preprocessing(self, img_encode):
        # decode the image
        img = tf.image.decode_jpeg(img_encode, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        # normalize image to -1 .. 1
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        # get multicrops depending on the image orientation
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
        bboxes = []

        for i in range(3):
            bbox = [
                i * is_h * offset, i * is_w * offset,
                tf.constant(self._cnn_input_size),
                tf.constant(self._cnn_input_size)
            ]

            img_crop = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
            img_crop = tf.expand_dims(img_crop, 0)

            img_array.append(img_crop)
            bboxes.append(bbox)

        return tf.concat(img_array, axis=0), bboxes

    def calc_output_dict(self, img_path):
        # feed forward batch of images in cnn and extract result
        activations_v, logits_v, bboxes_v = self._sess.run([self._activations, self.logits, self._bboxes],
                                                           feed_dict={self._image_path_placeholder: img_path})

        # softmax to get class probabilities with sum 1
        for crop in range(logits_v.shape[0]):
            logits_v[crop, :] = self._softmax(logits_v[crop, :])

        # fuse results of image crops using the maximum
        logits_v = np.mean(logits_v, axis=0)

        # assign logits to respective partitionings and get prediction (class with highest probability)
        partitioning_logits = []
        partitioning_pred = []
        partitioning_gps = []

        for p in range(self._num_partitionings):
            p_logits = logits_v[np.sum(self._classes_geo[0:p + 1]):np.sum(self._classes_geo[0:p + 2])]
            p_pred = p_logits.argsort()
            lat, lng = self._cell_centers[p][p_pred[-1]]

            partitioning_logits.append(p_logits)
            partitioning_pred.append(p_pred[-1])
            partitioning_gps.append([lat, lng])

        # get hierarchical multipartitioning results
        hierarchical_logits = partitioning_logits[-1]  # get logits from finest partitioning
        if self._num_partitionings > 1:
            for c in range(self._classes_geo[-1]):  # num_logits of finest partitioning
                for p in range(self._num_partitionings - 1):
                    hierarchical_logits[c] = hierarchical_logits[c] * partitioning_logits[p][self._cell_hierarchy[c][p]]

            pred = hierarchical_logits.argsort()
            lat, lng = self._cell_centers[self._num_partitionings - 1][pred[-1]]

            partitioning_logits.append(hierarchical_logits)
            partitioning_pred.append(pred[-1])
            partitioning_gps.append([lat, lng])

        # normalize all logits so that each vector has sum 1
        for p in range(len(partitioning_logits)):
            partitioning_logits[p] = partitioning_logits[p] / np.sum(partitioning_logits[p])

        self.output_dict = {
            'image_bboxes': bboxes_v,
            'activations': activations_v,
            'cell_probabilities': partitioning_logits,
            'predicted_cell_ids': partitioning_pred,
            'predicted_GPS_coords': partitioning_gps
        }

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

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
