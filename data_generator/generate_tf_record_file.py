# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        generate_tf_record_file.py
# Description: 生成tfRecord

"""
数据集制作过程：
(1) 每个文件夹放一个类别的图片
(2) 验证图片是否可用：check_data
(3) 生成tfRecord文件：generate_record_file
"""

from data_generator import img_util
from model.research.slim.datasets import convert_to_record_file


def check_data(data_dir, unqualified_dir):
    """
    验证图片是否可用
    Args:
        data_dir: 数据集目录
        unqualified_dir: 不可用数据存放目录

    Returns:

    """
    img_util.is_available_img(data_dir=data_dir, remove_dir=unqualified_dir)


def generate_record_file(dataset_dir, data_name, train_eval_split):
    """
    生成tfRecord文件
    Args:
        dataset_dir: 数据集目录
        data_name: 数据集名称
        train_eval_split: 评估集占比(默认0.2)

    Returns:

    """
    convert_to_record_file.run(dataset_dir=dataset_dir, data_name=data_name, train_eval_split=train_eval_split)


if __name__ == "__main__":
    _img_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/flowers'
    _unqualified_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/flowers_error"
    # check_data(data_dir=_img_dir, unqualified_dir=_unqualified_dir)
    generate_record_file(dataset_dir=_img_dir, data_name="flowers", train_eval_split=0.1)


