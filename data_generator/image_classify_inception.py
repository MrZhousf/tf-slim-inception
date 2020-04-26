# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        image_classify_inception.py
# Description: 生成slim数据格式的tfRecord
# 每个文件夹放一个类别图片即可
# 数据制作： dataset_factory.py flowers.py download_and_cover_flowers.py

# from datasets import download_and_convert_renbao
# from datasets import download_and_convert_car_color
# from datasets import download_and_convert_singapore
# from datasets import download_and_convert_windscreen
import os
import img_util
import file_util


def convert_record(convert, img_dir, data_name):
    """
    生成slim数据格式的tfRecord与label.txt标签文件
    :param convert:
    :param img_dir:
    :param data_name:
    :return:
    """
    convert.run(img_dir, data_name)


def check_data(img_dir, unqualified_dir):
    if not os.path.exists(unqualified_dir):
        os.makedirs(unqualified_dir)
    total = 0
    unqualified = 0
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            total += 1
            print(img)
            if not img_util.is_img(img):
                file_util.move_file(os.path.join(root, img), unqualified_dir)
                unqualified += 1
    print("total={0}, unqualified={1}".format(total, unqualified))


if __name__ == "__main__":
    _img_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/sorter'
    _unqualified_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/sorter_unqualified"
    # check_data(img_dir=_img_dir, unqualified_dir=_unqualified_dir)
    from datasets import download_and_convert_sorter
    convert_record(img_dir=_img_dir, data_name="sorter", convert=download_and_convert_sorter)


