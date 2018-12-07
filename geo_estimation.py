import argparse
import csv
import json
import numpy as np
import os
import re
import s2sphere as s2
import sys
from scipy.misc import imresize as imresize
from scipy.misc import imread as imread
import matplotlib.pyplot as plt

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
        self._image_ph = tf.placeholder(shape=[3, self._cnn_input_size, self._cnn_input_size, 3], dtype=tf.float32)
        config = tf.ConfigProto()
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

        var_list = {
            re.sub('^' + self.scope.name + '/', '', x.name)[:-2]: x for x in tf.global_variables(self.scope.name)
        }

        # restore weights
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, model_file)

        # get activations from last conv layer and output weights in order to calculate class activation maps
        self.activations = tf.get_default_graph().get_tensor_by_name(self.scope.name + '_1/resnet_v2_101/activations:0')
        activation_weights = tf.get_default_graph().get_tensor_by_name(self.scope.name +
                                                                       '/classifier_geo/logits/weights:0')

        # activation_bias = tf.get_default_graph().get_tensor_by_name(self.scope.name + '/classifier_geo/logits/biases:0')
        # activation_weights = activation_weights + activation_bias

        # store activation_weights of model
        activation_weights_v = self.sess.run(activation_weights)

        p_activation_weights = []
        for p in range(self._num_partitionings):
            p_activation_weights.append(
                activation_weights_v[:, :, :,
                                     np.sum(self._classes_geo[0:p + 1]):np.sum(self._classes_geo[0:p + 2])])

        self.network_dict = {
            'activation_weights': p_activation_weights,
            'partitionings': partitionings,
            'classes_geo': self._classes_geo[1:],  # 0 was appended for simplification
            'cell_centers': self._cell_centers,
            'scope': self.scope.name
        }

    def calc_output_dict(self, img_path, show_cam=True):
        img_crops, bboxes = self._img_preprocessing(img_path)

        # feed forward batch of images in cnn and extract result
        activations_v, logits_v = self.sess.run([self.activations, self.logits], feed_dict={self._image_ph: img_crops})

        # softmax to get class probabilities with sum 1 for each partition in each crop
        for crop in range(logits_v.shape[0]):
            for p in range(self._num_partitionings):
                idx_s = np.sum(self._classes_geo[0:p + 1])
                idx_e = np.sum(self._classes_geo[0:p + 2])
                logits_v[crop, idx_s:idx_e] = self._softmax(logits_v[0, idx_s:idx_e])

        # fuse results of image crops using the maximum
        logits_v = np.max(logits_v, axis=0)  # NOTE: sum of each partitioning will be > 1 while using mas

        # assign logits to respective partitionings and get predictions (class with highest probability)
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

        # return results
        self.output_dict = {
            'bboxes': bboxes,
            'predicted_cell_ids': partitioning_pred,
            'predicted_GPS_coords': partitioning_gps,
            'activations': activations_v,
            'logits': partitioning_logits,
        }

        return 0

    def calc_class_activation_map(self, class_idx, partition_idx):
        # get weights of specified class and partition
        class_activation_weights = self.network_dict['activation_weights'][partition_idx][0, 0, :, class_idx]

        # get dimensions
        num_crops, h, w, num_features = self.output_dict['activations'].shape
        img_size = self.output_dict['bboxes'][0][-1]
        r = h / img_size

        # create output variables
        cam = np.zeros(shape=[
            int(self.output_dict['bboxes'][-1][0] * r + 0.5) + w,
            int(self.output_dict['bboxes'][-1][1] * r + 0.5) + h
        ])

        num_activations = np.zeros(shape=[
            int(self.output_dict['bboxes'][-1][0] * r + 0.5) + w,
            int(self.output_dict['bboxes'][-1][1] * r + 0.5) + h
        ])

        # generate class activation map for each image crop
        for crop_idx in range(num_crops):
            # get activation map of current crop
            crop_activations = self.output_dict['activations'][crop_idx, :, :, :]
            crop_activation_map = class_activation_weights.dot(crop_activations.reshape((num_features, h * w)))
            crop_activation_map = crop_activation_map.reshape(h, w)

            # translate bbox coordinates from original image size to feature size
            feature_bbox = []
            for entry in self.output_dict['bboxes'][crop_idx]:
                feature_bbox.append(int(entry * r + 0.5))

            # store class activation map of the crop
            cam[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += crop_activation_map
            num_activations[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += 1

        # NOTE: prevent division by 0, if the whole image is not covered with all crops [max_dim > 3 * min_dim]
        num_activations[num_activations == 0] = 1

        # normalize class activation map
        cam /= num_activations
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = np.asarray(cam * 255 + 0.5, dtype=np.uint8)

        return cam

    def draw_class_activation_map(self, img, class_activation_map, img_alpha=0.6, size=None):
        # resize input images
        if size is not None:
            r = size / np.minimum(img.shape[0], img.shape[1])
            img = imresize(img, size=[int(r * img.shape[0] + 0.5), int(r * img.shape[1] + 0.5)])

        class_activation_map = imresize(class_activation_map, size=[img.shape[0], img.shape[1]])

        # create rgb overlay
        cm = plt.get_cmap('jet')
        cam_ovlr = cm(class_activation_map)

        # normalize to 0..1 and convert to grayscale
        img = img / 255.0
        img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # create heatmap composite
        cam_heatmap = img_alpha * np.expand_dims(img_gray, axis=-1) + (1 - img_alpha) * cam_ovlr[:, :, 0:3]

        # visualize
        plt.imshow(cam_heatmap)
        plt.show()

    def _img_preprocessing(self, img_path):
        # read image
        img = imread(img_path, mode='RGB')

        # crop image into three evenly sampled crops
        img_crops, bboxes = self._crop_image(img)

        # normalize to -1 .. 1
        img_crops = img_crops / 255.0
        img_crops -= 0.5
        img_crops *= 2.0

        return img_crops, bboxes

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _crop_image(self, img):
        height = img.shape[0]
        width = img.shape[1]

        # get minimum and maximum coordinate
        max_side_len = np.maximum(width, height)
        min_side_len = np.minimum(width, height)
        is_w, is_h = (0, 1) if (width < height) else (1, 0)

        # resize image
        ratio = self._cnn_input_size / min_side_len
        offset = (int(max_side_len * ratio + 0.5) - self._cnn_input_size) // 2
        img_resized = imresize(img, size=[int(height * ratio + 0.5), int(width * ratio + 0.5)])

        # get crops according to image orientation
        img_array = []
        bboxes = []
        for i in range(3):
            bbox = [i * is_h * offset, i * is_w * offset, self._cnn_input_size, self._cnn_input_size]
            img_crop = img_resized[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]]
            img_crop = np.expand_dims(img_crop, axis=0)

            bboxes.append(bbox)
            img_array.append(img_crop)

        return np.concatenate(img_array, axis=0), bboxes

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
