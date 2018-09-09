from . import *
import tensorflow as tf
import numpy as np  #
import logging

slim = tf.contrib.slim

architectures_dict = {
    'vgg_16': {
        'model': vgg.vgg_16,
        'exclude': ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'],
        'scope_name': 'vgg_16',
        'arg_scope': vgg.vgg_arg_scope
    },
    'vgg_19': {
        'model': vgg.vgg_19,
        'exclude': ['vgg_19/fc6', 'vgg_19/fc7', 'vgg_19/fc8'],
        'scope_name': 'vgg_19',
        'arg_scope': vgg.vgg_arg_scope
    },
    'resnet_v2_50': {
        'model': resnet_v2.resnet_v2_50,
        'exclude': ['resnet_v2_101/logits', 'classifier_geo/logits', 'classifier_scenes/logits'],
        'scope_name': 'resnet_v2_50',
        'arg_scope': resnet_v2.resnet_arg_scope
    },
    'resnet_v2_101': {
        'model': resnet_v2.resnet_v2_101,
        'exclude': ['resnet_v2_101/logits', 'classifier_geo/logits', 'classifier_scenes/logits'],
        'scope_name': 'resnet_v2_101',
        'arg_scope': resnet_v2.resnet_arg_scope
    },
    'resnet_v2_101_fine': {
        'model': resnet_v2.resnet_v2_101,
        'exclude': [],
        'scope_name': 'resnet_v2_101',
        'arg_scope': resnet_v2.resnet_arg_scope
    },
    'resnet_v2_152': {
        'model': resnet_v2.resnet_v2_152,
        'exclude': ['resnet_v2_152/logits'],
        'scope_name': 'resnet_v2_152',
        'arg_scope': resnet_v2.resnet_arg_scope
    }
}


def create_model(architecture, inputs, num_classes, is_training=True, scope=None, reuse=None):
    if architecture not in architectures_dict:
        logging.error('CNN model not found')
        return None

    model_config = architectures_dict[architecture]

    model = model_config['model']
    arg_scope = model_config['arg_scope']
    #with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    if scope is None:
        scope = model_config['scope_name']
    if reuse is not None:
        tf.get_variable_scope().reuse_variables()
    with slim.arg_scope(arg_scope()):
        logits, endpoints = model(inputs, num_classes=num_classes, is_training=is_training)

    # logits = tf.squeeze(logits)

    return logits, endpoints


def model_weight_excludes(architecture):
    if architecture not in architectures_dict:
        logging.error('CNN model not found')
        return []

    model_config = architectures_dict[architecture]

    if 'exclude' not in model_config:
        return []

    return model_config['exclude']


def model_trainable_variables(architecture):
    if architecture not in architectures_dict:
        logging.error('CNN model not found')
        return []

    model_config = architectures_dict[architecture]

    if 'trainable' not in model_config:
        return []

    return model_config['trainable']
