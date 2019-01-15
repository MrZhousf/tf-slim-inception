# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-27
# File:        classify_image.py
# Description:  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf


class EvalNode(object):

    def __init__(self, image_file, model_path, label_path,
          image_size, num_top_predictions, output_node_name):
        self.image_file = image_file
        self.model_path = model_path
        self.label_path = label_path
        self.image_size = image_size
        self.num_top_predictions = num_top_predictions
        self.output_node_name = output_node_name
        self.node_id_to_name = None

    def load(self):
        node_id_to_name = {}
        with open(self.label_path) as f:
            for index, line in enumerate(f):
                node_id_to_name[index] = line.strip()
        self.node_id_to_name = node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_id_to_name:
            return ''
        return self.node_id_to_name[node_id]


def create_graph(node_lookup):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(node_lookup.model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def infer(image_file, model_path, label_path,
          image_size=299, num_top_predictions=5, output_node_name='InceptionV4/Logits/Predictions:0'):
    with tf.Graph().as_default():
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        image_data = tf.image.decode_jpeg(image_data)
        image_data = preprocess_for_eval(image_data, image_size, image_size)
        image_data = tf.expand_dims(image_data, 0)
        with tf.Session() as sess:
            image_data = sess.run(image_data)
    node_lookup = EvalNode(image_file, model_path, label_path, image_size, num_top_predictions, output_node_name)
    create_graph(node_lookup)
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name(node_lookup.output_node_name)
        predictions = sess.run(softmax_tensor, {'input:0': image_data})
        predictions = np.squeeze(predictions)
        node_lookup.load()
        top_k = predictions.argsort()[-node_lookup.num_top_predictions:][::-1]
        result = {}
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            result[human_string] = score
        return sorted(result.items(), key=lambda d: d[1], reverse=True)
