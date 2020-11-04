# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        image_classify_inception.py
# Description: 生成slim数据格式的tfRecord
# 每个文件夹放一个类别图片即可
# 数据制作： dataset_factory.py flowers.py download_and_cover_flowers.py

from data_generator import img_util
from model.research.slim.datasets import download_and_convert_certificate


def check_data(img_dir, unqualified_dir):
    img_util.is_available_img(data_dir=img_dir, remove_dir=unqualified_dir)


if __name__ == "__main__":
    _img_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/test/certificate/original'
    _unqualified_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/test/certificate/original_error"
    # check_data(img_dir=_img_dir, unqualified_dir=_unqualified_dir)
    download_and_convert_certificate.run(dataset_dir=_img_dir)


