# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2019-12-24
# File:        prediction.py
# Description:
# coding: utf-8
import tensorflow as tf
import os
import time
import numpy as np


class Prediction(object):
    INCEPTION_V3 = 'InceptionV3/Predictions/Reshape_1:0'
    INCEPTION_V4 = 'InceptionV4/Logits/Predictions:0'
    INCEPTION_RESNET_V2 = 'InceptionResnetV2/Logits/Predictions:0'

    def __init__(self,
                 pb_file,
                 label_file,
                 image_size=299,
                 num_top_predictions=5,
                 model_type=INCEPTION_V3,
                 gpu_assigned="1",
                 per_process_gpu_memory_fraction=0.2):
        self.pb_file = pb_file
        self.label_file = label_file
        self.image_size = image_size
        self.num_top_predictions = num_top_predictions
        self.model_type = model_type
        self.node_id_to_name = self.load()
        with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
            graph = tf.Graph()
            with graph.as_default():
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.config = tf.ConfigProto()
                self.config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
                if per_process_gpu_memory_fraction == 0.0:
                    self.config.gpu_options.allow_growth = True
                else:
                    self.config.gpu_options.allow_growth = False
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_assigned
                self.sess = tf.Session(config=self.config, graph=graph)
                self.softmax_tensor = self.sess.graph.get_tensor_by_name(self.model_type)

    def load(self):
        node_id_to_name = {}
        with open(self.label_file) as f:
            for index, line in enumerate(f):
                node_id_to_name[index] = line.strip()
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_id_to_name:
            return ''
        return self.node_id_to_name[node_id]

    @staticmethod
    def deal_image(image,
                   height,
                   width,
                   central_fraction=0,
                   scope=None):
        """
        center crop
        :param image:
        :param height:
        :param width:
        :param central_fraction: 0 not crop; 0.875 crop
        :param scope:
        :return:
        """
        with tf.name_scope(scope, 'eval_image', [image, height, width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            if central_fraction != 0:
                image = tf.image.central_crop(image, central_fraction=central_fraction)
            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image,
                                                 [height, width],
                                                 align_corners=False)
                image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image

    def infer(self, image_file):
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        with tf.Graph().as_default():
            image_data = tf.image.decode_jpeg(image_data)
            image_data = self.deal_image(image_data, self.image_size, self.image_size)
            image_data = tf.expand_dims(image_data, 0)
            with tf.Session() as sess:
                image_data = sess.run(image_data)
        predictions = self.sess.run(self.softmax_tensor, {'input:0': image_data})
        predictions = np.squeeze(predictions)
        result_ = {}
        top_k = predictions.argsort()[-self.num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = self.id_to_string(node_id)
            score = predictions[node_id]
            result_[human_string] = score
        return sorted(result_.items(), key=lambda d: d[1], reverse=True)



if __name__ == "__main__":
    pb = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/workspace/zhousf_projects/tf-slim-inception-master/my_models/inception_v3/sorter/export/10000/frozen_inference_graph.pb"
    clz = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/workspace/zhousf_projects/tf-slim-inception-master/my_models/inception_v3/sorter/labels.txt"
    img_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/sorter_eval/vin"
    pre = Prediction(pb_file=pb, label_file=clz, num_top_predictions=1, model_type=Prediction.INCEPTION_V3)
    total = 0
    true_num = 0
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            start = time.time()
            current_file = os.path.join(root, file)
            class_name = os.path.basename(os.path.dirname(current_file))
            result = pre.infer(current_file)
            if result[0][0] == class_name:
                true_num += 1
            total += 1
            print('{0} Time cost：{1}s {2}'.format(total, time.time() - start, result))
    rate = (true_num / total) * 100
    print("accuracy={0}/{1}={2}%".format(true_num, total, rate))



