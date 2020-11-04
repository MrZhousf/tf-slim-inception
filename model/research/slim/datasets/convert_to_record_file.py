# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import tensorflow as tf

from datasets import dataset_utils
from prettytable import PrettyTable
import json

# 生成record文件的个数
_NUM_SHARDS = 4


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, trian_eval_split):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    cls_img_num = {}
    photo_filenames_train = []
    photo_filenames_eval = []
    describe = {}
    for directory in directories:
        class_name = directory.split("/")[-1]
        img_num = len(os.listdir(directory))
        eval_num = int(img_num * trian_eval_split)
        cls_img_num[class_name] = eval_num
        describe[class_name] = {"train_num": img_num - eval_num, "eval_num": eval_num,
                                "total": img_num}
        current_cls_img = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            current_cls_img.append(path)
        photo_filenames_train.extend(current_cls_img[eval_num:])
        photo_filenames_eval.extend(current_cls_img[:eval_num])
    return photo_filenames_train, photo_filenames_eval, sorted(class_names), describe


def _get_dataset_filename(dataset_dir, split_name, shard_id, data_name):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        data_name, split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, data_name):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    error_img = []
    valid_img_num = 0
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, data_name)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i + 1, len(filenames), shard_id))
                            sys.stdout.flush()

                            # Read the filename:
                            image_data = tf.gfile.GFile(filenames[i], 'rb').read()
                            height, width = image_reader.read_image_dims(sess, image_data)

                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            class_id = class_names_to_ids[class_name]

                            example = dataset_utils.image_to_tfexample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
                            valid_img_num += 1
                        except Exception as e:
                            error_img.append(filenames[i])
    sys.stdout.write('\n')
    sys.stdout.flush()
    if len(error_img) > 0:
        print("ERROR IMAGE:")
    for img in error_img:
        print(img)
    return valid_img_num


def _dataset_exists(dataset_dir, data_name):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id, data_name)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir, data_name, train_eval_split=0.2):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      data_name: The name of dataset.
      train_eval_split: Proportion of evaluation set.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir, data_name):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    training_filenames, validation_filenames, class_names, describe = _get_filenames_and_classes(
        dataset_dir, train_eval_split)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # First, convert the training and validation sets.
    valid_train_img_num = _convert_dataset('train', training_filenames, class_names_to_ids,
                                           dataset_dir, data_name)
    valid_eval_img_num = _convert_dataset('validation', validation_filenames, class_names_to_ids,
                                          dataset_dir, data_name)
    num_file = os.path.join(dataset_dir, "train_eval_num.txt")
    table = PrettyTable(["class_name", "train_num", "eval_num", "total"])
    for cls_des in describe:
        table.add_row(
            [cls_des, describe.get(cls_des).get("train_num"),
             describe.get(cls_des).get("eval_num"),
             describe.get(cls_des).get("total")])
    table.align["class_name"] = "l"  # l=left
    js = json.dumps({"train": valid_train_img_num, "validation": valid_eval_img_num, "classes_num": len(class_names)})
    with tf.gfile.Open(num_file, 'w') as f:
        f.write(js)
        f.write('\n')
        f.write('\n')
        f.write('%s\n' % str(table))

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print(table)
    print('\nFinished converting the dataset!')
