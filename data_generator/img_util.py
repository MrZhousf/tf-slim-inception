# -*- coding: utf-8 -*-
# Author:      zhousf
# Date:        2019-06-18
# File:        img_util.py
# Description: 图片处理: 图片增强、旋转、镜像
import os, shutil
import cv2
from data_generator import file_util
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import json
import base64
import imghdr


class ImgOperator:
    """
    水平镜像、垂直镜像、旋转90度、旋转270度
    """
    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    # 增加噪点
    def add_noise(self, img_file_path):
        img = cv2.imread(img_file_path)
        noise_num = 10000
        for k in range(0, noise_num):
            xi = int(np.random.uniform(0, img.shape[1]))
            xj = int(np.random.uniform(0, img.shape[0]))
            if img.ndim == 2:
                img[xj, xi] = 255
            elif img.ndim == 3:
                img[xj, xi, 0] = 25
                img[xj, xi, 1] = 20
                img[xj, xi, 2] = 20
        cv2.imwrite(img_file_path, img)

    # 逆时针旋转90度
    def rotate_90(self, img_path):
        image = Image.open(img_path)
        width, height = image.size
        img_pixel = image.load()
        rotate = Image.new(image.mode, (height, width), "white")
        for i in range(height):
            for j in range(width):
                pixel = img_pixel[j, i]
                rotate.putpixel((i, j), pixel)
        return rotate

    # 逆时针旋转270度
    def rotate_270(self, img_path):
        image = Image.open(img_path)
        width, height = image.size
        img_pixel = image.load()
        rotate = Image.new(image.mode, (height, width), "white")
        for i in range(height):
            for j in range(width):
                pixel = img_pixel[j, height - 1 - i]
                rotate.putpixel((i, j), pixel)
        return rotate

    '''
    生成水平镜像、垂直镜像、水平垂直镜像、旋转90度、旋转270度
    共5张
    '''
    def auto_deal(self):
        jpg_list = os.listdir(self.src_dir)
        total_num = 0
        for i in range(0, len(jpg_list)):
            jpg_file = jpg_list[i]
            if jpg_file.endswith('.jpg'):
                if self.__deal_img(self.src_dir, jpg_file, '_rotate_90', self.TYPE_ROTATE_90, add_noise=True) is not None:
                    total_num += 1
                if self.__deal_img(self.src_dir, jpg_file, '_rotate_270', self.TYPE_ROTATE_270, add_noise=True) is not None:
                    total_num += 1
                if self.__deal_img(self.src_dir, jpg_file, '_mirror_h', self.TYPE_MIRROR_HORIZONTAL, add_noise=True) is not None:
                    total_num += 1
                vf = self.__deal_img(self.src_dir,jpg_file, '_mirror_v', self.TYPE_MIRROR_VERTICAL, add_noise=True)
                if vf is None:
                    continue
                total_num += 1
                # hvf = os.path.basename(vf)
                # if self.__deal_img(self.dst_dir, hvf, '_mirror_h', self.TYPE_MIRROR_HORIZONTAL, add_noise=True) is not None:
                #     total_num += 1
                print(str(total_num))
        print('共生成新样本：' + str(total_num) + '项')

    # 水平镜像
    def mirror_horizontal(self,add_noise=False):
        tag = '_mirror_horizontal'
        jpg_list = os.listdir(self.src_dir)
        total_num = 0
        for i in range(0,len(jpg_list)):
            jpg_file = jpg_list[i]
            if jpg_file.endswith('.jpg'):
                self.__deal_img(self.src_dir, jpg_file, tag, self.TYPE_MIRROR_HORIZONTAL, add_noise)
                total_num += 1
        print('共生成：'+str(total_num)+'项')

    # 垂直镜像
    def mirror_vertical(self,add_noise=False):
        tag = '_mirror_vertical'
        jpg_list = os.listdir(self.src_dir)
        total_num = 0
        for i in range(0, len(jpg_list)):
            jpg_file = jpg_list[i]
            if jpg_file.endswith('.jpg'):
                self.__deal_img(self.src_dir, jpg_file,tag,self.TYPE_MIRROR_VERTICAL,add_noise)
                total_num += 1
        print('共生成镜像：' + str(total_num) + '项')

    TYPE_MIRROR_HORIZONTAL = 1
    TYPE_MIRROR_VERTICAL = 2
    TYPE_ROTATE_90 = 3
    TYPE_ROTATE_270 = 4

    '''
    deal_type:
    TYPE_MIRROR_HORIZONTAL-水平镜像
    TYPE_MIRROR_VERTICAL-垂直镜像
    TYPE_ROTATE_90-逆时针旋转90度
    TYPE_ROTATE_270-逆时针旋转270度
    '''
    def __deal_img(self,src_dir,jpg_file,tag,deal_type=TYPE_MIRROR_HORIZONTAL,add_noise=False):
        jpg_path = os.path.join(src_dir, jpg_file)
        new_img_file = ''
        if jpg_file.endswith('.jpg'):
            file_name, ext = os.path.basename(jpg_path).split('.jpg')
            new_img_file = os.path.join(self.dst_dir, file_name) + tag + '.jpg'
            new_json_file = os.path.join(self.dst_dir, file_name) + tag + '.json'
            json_file = os.path.join(src_dir, file_name) + '.json'
            original_img = Image.open(jpg_path)
            width, height = original_img.size
            global deal_img
            if deal_type == self.TYPE_MIRROR_HORIZONTAL:
                deal_img = original_img.transpose(Image.FLIP_LEFT_RIGHT)
                deal_img.save(new_img_file)
            elif deal_type == self.TYPE_MIRROR_VERTICAL:
                deal_img = original_img.transpose(Image.FLIP_TOP_BOTTOM)
                deal_img.save(new_img_file)
            elif deal_type == self.TYPE_ROTATE_90:
                deal_img = self.rotate_90(jpg_path)
                deal_img.save(new_img_file)
            elif deal_type == self.TYPE_ROTATE_270:
                deal_img = self.rotate_270(jpg_path)
                deal_img.save(new_img_file)
            if add_noise:
                self.add_noise(new_img_file)
            with open(new_img_file, 'rb') as f:
                image_data = base64.b64encode(f.read())
                with open(json_file, 'r') as load_f:
                    load_dict = json.load(load_f)
                    shapes = load_dict['shapes']
                    load_dict['imageData'] = image_data
                    load_dict['imagePath'] = file_name + tag + '.jpg'
                    print(file_name,deal_img.width,deal_img.height)
                    for i in range(0, len(shapes)):
                        points = shapes[i]['points']
                        for n in range(0, len(points)):
                            if deal_type == self.TYPE_MIRROR_HORIZONTAL:
                                points[n][0] = width - 1 - points[n][0]
                                points[n][1] = points[n][1]
                            elif deal_type == self.TYPE_MIRROR_VERTICAL:
                                points[n][0] = points[n][0]
                                points[n][1] = height - 1 - points[n][1]
                            elif deal_type == self.TYPE_ROTATE_90:
                                tmp = points[n][0]
                                points[n][0] = points[n][1]
                                points[n][1] = tmp
                                load_dict['imgWidth'] = deal_img.height
                                load_dict['imgHeight'] = deal_img.width
                            elif deal_type == self.TYPE_ROTATE_270:
                                tmp = points[n][0]
                                points[n][0] = height - 1 - points[n][1]
                                points[n][1] = tmp
                                load_dict['imgWidth'] = deal_img.height
                                load_dict['imgHeight'] = deal_img.width
                    json.dump(load_dict, open(new_json_file, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
                load_f.close()
            f.close()
        return new_img_file


def fetch_max_width_height(img_dir, ext_list={'.JPG', '.jpg', '.png'}):
    """
    获取img_dir目录中图片最大的宽和高的文件
    :param img_dir: 图片目录
    :param ext_list: 图片扩展名
    :return:
    """
    width_max = [0, 0, 0]
    width_max_file = ''
    height_max = [0, 0, 0]
    height_max_file = ''
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            ext = file_util.file_extension(os.path.join(root, f))
            if ext in ext_list:
                img = cv2.imread(os.path.join(root, f))
                sp = img.shape
                height = sp[0]
                width = sp[1]
                channel = sp[2]
                if width > width_max[0]:
                    width_max[0] = width
                    width_max[1] = height
                    width_max[2] = channel
                    width_max_file = os.path.join(root, f)
                if height > height_max[1]:
                    height_max[0] = width
                    height_max[1] = height
                    height_max[2] = channel
                    height_max_file = os.path.join(root, f)
                print ('width: %d height: %d number: %d' % (width, height, channel))
    print('宽度最大的图片：' + str(width_max) + ',' + width_max_file)
    print('高度最大的图片：' + str(height_max) + ',' + height_max_file)


def move_limit_width_height(img_dir, out_limit_dir, limit_width, limit_height, ext_list={'.JPG', '.jpg', '.png'}):
    """
    将img_dir中所有宽或高大于限制宽或高的图片移动到out_limit_dir目录中
    :param img_dir:
    :param out_limit_dir:
    :param limit_width:
    :param limit_height:
    :param ext_list:
    :return:
    """
    if not os.path.exists(out_limit_dir):
        os.makedirs(out_limit_dir)
    total = 0
    limit = 0
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            ext = file_util.file_extension(os.path.join(root, f))
            if ext in ext_list:
                total += 1
                img = cv2.imread(os.path.join(root, f))
                if img is None:
                    file_util.move_file(os.path.join(root, f), out_limit_dir)
                    json_f = file_util.file_basename(f)
                    json_f += '.json'
                    file_util.move_file(os.path.join(root, json_f), out_limit_dir)
                    continue
                sp = img.shape
                height = sp[0]
                width = sp[1]
                channel = sp[2]
                if width > limit_width:
                    file_util.move_file(os.path.join(root, f), out_limit_dir)
                    json_f = file_util.file_basename(f)
                    json_f += '.json'
                    file_util.move_file(os.path.join(root, json_f), out_limit_dir)
                    print('width: %d height: %d channel: %d' % (width, height, channel))
                    limit += 1
                    continue
                if height > limit_height:
                    file_util.move_file(os.path.join(root, f), out_limit_dir)
                    json_f = file_util.file_basename(f)
                    json_f += '.json'
                    file_util.move_file(os.path.join(root, json_f), out_limit_dir)
                    print('width: %d height: %d channel: %d' % (width, height, channel))
                    limit += 1
                    continue
    print('total=' + str(total) + ',limit=' + str(limit))


def img2gray(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(imagePath, gray)


def gray2rgb_single(img_path):
    src = cv2.imread(img_path, 0)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(img_path, src_RGB)


def gray2rgb(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            ext = file_util.file_extension(f)
            if ext in {'.jpg', '.JPG'}:
                path = os.path.join(root, f)
                print(path)


def show(img_path):
    new = cv2.imread(img_path)
    cv2.imshow(img_path, new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _fetch_img(img):
    if isinstance(img, Image.Image):
        return img
    else:
        return Image.open(img)


def _deal_enhance(enh, rate, show_img=True):
    img_enhanced = enh.enhance(rate)
    if show_img:
        img_enhanced.show()
    else:
        return img_enhanced


def enhance_brightness(img, rate=1.5, show_img=True):
    """
    亮度增强
    :param img:
    :param rate: 亮度增强率
    :param show_img:
    :return:
    """
    img = _fetch_img(img)
    enh = ImageEnhance.Brightness(img)
    return _deal_enhance(enh,rate,show_img)


def enhance_color(img, rate=1.5, show_img=True):
    """
    颜色增强
    :param img:
    :param rate: 颜色增强率
    :param show_img:
    :return:
    """
    img = _fetch_img(img)
    enh = ImageEnhance.Color(img)
    return _deal_enhance(enh, rate, show_img)


def enhance_contrast(img, rate=1.2, show_img=True):
    """
    对比度增强
    :param img:
    :param rate: 对比度增强率
    :param show_img:
    :return:
    """
    img = _fetch_img(img)
    enh = ImageEnhance.Contrast(img)
    return _deal_enhance(enh, rate, show_img)


def enhance_sharpness(img, rate=6.0, show_img=True):
    """
    锐度增强
    :param img:
    :param rate: 锐度增强率
    :param show_img:
    :return:
    """
    img = _fetch_img(img)
    enh = ImageEnhance.Sharpness(img)
    return _deal_enhance(enh, rate, show_img)


def enhance_overwrite(img_path):
    """
    图片增强并覆盖原文件
    :param img_path:
    :return:
    """
    img = enhance_color(img=img_path, rate=1.1, show_img=False)
    img = enhance_contrast(img=img, rate=1.1, show_img=False)
    img = enhance_sharpness(img=img, rate=3.0, show_img=False)
    img.save(img_path)


def enhance_img(img_path):
    """
    图片增强并覆盖原文件
    :param img_path:
    :return:
    """
    img = enhance_color(img=img_path, rate=1.1, show_img=False)
    img = enhance_contrast(img=img, rate=1.1, show_img=False)
    img = enhance_sharpness(img=img, rate=3.0, show_img=False)
    return img


def enhance(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    total = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.JPG'):
                print(f)
                name, _ = f.split('.')
                json_f = os.path.join(root, name+'.json')
                img = os.path.join(root, f)
                # 亮度增强会影响损伤效果
                # img = enhance_brightness(img=img, rate=1.1, show_img=False)
                img = enhance_color(img=img, rate=1.1, show_img=False)
                img = enhance_contrast(img=img, rate=1.1, show_img=False)
                img = enhance_sharpness(img=img, rate=3.0, show_img=False)
                save_img = os.path.join(dst_dir, f)
                img.save(save_img)
                if os.path.exists(json_f):
                    shutil.copy(json_f, dst_dir)
                total += 1
    print('total=%d' % total)


def compress_img(src_dir, quality=20):
    total = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            name, ext = f.split('.')
            if ext not in ['jpg', 'JPG']:
                continue
            img_file = os.path.join(root, f)
            print(img_file)
            img = Image.open(img_file)
            img.save(img_file, quality=quality)
            total += 1

    print('total='+str(total))


def is_img(img_file):
    if imghdr.what(img_file):
        return True
    return False


def save_img_form(image, save_path):
    """
    保存form表单中的图片
    :param image: request.files.get("image")
    :param save_path: 不带图片拓展名称(.jpg)
    :return:
    """
    img_type = image.filename.split(".")[-1]
    save_full_path = save_path+"."+img_type
    image.save(save_full_path)
    if not os.path.exists(save_full_path):
        return False, "save {0} failed".format(image.name)
    size = os.path.getsize(save_full_path)
    if size == 0:
        return False, "image content length is zero"
    img = Image.open(save_full_path)
    img_array = np.array(img)
    if img_array.shape[2] != 3:
        img = img.convert("RGB")
        img.save(save_full_path)
        if not os.path.exists(save_full_path):
            return False, "convert image to RGB failed".format(image.name)
    return True, save_full_path


def check_img_available_form(image, img_types=None):
    """
    检查form表单中的图片格式是否支持
    :param image:
    :param img_types:图片格式
    :return:
    """
    img_type = image.filename.split(".")[-1]
    img_type = img_type.lower()
    if img_types is None:
        img_types = ["jpg", "jpeg", "png"]
    if img_type not in img_types:
        return False
    return True


def is_available_img(data_dir, remove_dir):
    """
    检测图片是否可用
    :param data_dir: 图片目录
    :param remove_dir: 不可用文件保存目录
    :return:
    """
    if os.path.exists(remove_dir):
        file_util.delete_dir(remove_dir)
    os.makedirs(remove_dir)
    total = 0
    unqualified = 0
    for root, dirs, files in os.walk(data_dir):
        for img in files:
            total += 1
            print(total)
            current_file = os.path.join(root, img)
            if not is_img(current_file):
                file_util.move_file(current_file, remove_dir)
                unqualified += 1
                continue
            if not is_rgb(current_file):
                file_util.move_file(current_file, remove_dir)
                unqualified += 1
                continue
            # 判定jpg是否包含结束字段
            with open(current_file, 'rb') as f:
                f.seek(-2, 2)
                buf = f.read()
                f.close()
                if buf != b'\xff\xd9':
                    file_util.move_file(current_file, remove_dir)
                    unqualified += 1
                    continue
    print("total={0}, remove={1}".format(total, unqualified))


def is_rgb(img_file):
    try:
        img = Image.open(img_file)
        img_array = np.array(img)
        if img_array.shape[2] != 3:
            return False
        pass
    except Exception:
        return False
    return True


def check_img_rgb(src_dir, dst_dir):
    total = 0
    not_rgb = 0
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not is_img(file):
                continue
            f = os.path.join(root, file)
            if not is_rgb(f):
                print(file)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.move(f, os.path.join(dst_dir, file))
                not_rgb += 1
            total += 1
            print(total)
    print("共{0}项，{1}项不是rgb".format(total, not_rgb))


def mod_img_width_height(src_dir, dst_dir, width, height):
    """
    修改图片的宽高
    :param src_dir:
    :param dst_dir:
    :param width:
    :param height:
    :return:
    """
    if dst_dir is None:
        dst_dir = src_dir
    else:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not is_img(file):
                continue
            im = Image.open(os.path.join(root, file))
            if im.size[0] > width or im.size[1] > height:
                print(im.format, im.size, im.mode)
                im.thumbnail((width, height))
                im.save(os.path.join(dst_dir, file), 'JPEG')


def find_repeat_img(data_pool, data_dir, repeat_dir):
    """
    查询重复的图片
    :param data_pool:图片池
    :param data_dir: 待处理图片
    :param repeat_dir: 重复图片保存目录
    :return:
    """
    pool = []
    for root, dirs, files in os.walk(data_pool):
        for file in files:
            name, _ = file.split(".")
            if name not in pool:
                pool.append(name)
    if not os.path.exists(repeat_dir):
        os.makedirs(repeat_dir)
    total = 0
    repeat = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            name, _ = file.split(".")
            total += 1
            if name in pool:
                repeat += 1
                file_util.move_file(os.path.join(root, file), repeat_dir)
    print("total={0}, repeat={1}".format(total, repeat))
    pass


if __name__ == '__main__':
    src_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/part/middle'
    dst_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/part/error'
    data_pool = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/windscreen/singapore/data"
    data_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/windscreen/jy/damage"
    repeat_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/windscreen/repeat"
    # find_repeat_img(data_pool, data_dir, repeat_dir)
    # mod_img_width_height(src_dir, dst_dir=None, width=1920, height=1920)
    # enhance(src_dir, dst_dir)
    # check_img_rgb(src_dir, dst_dir)
    # a = cv2.imread('/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/1/MT1042898-001/1.JPG')
    # print(cv2.imwrite('/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/1/MT1042898-001/1_1.JPG', a))
    # compress_img('/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/test/')
    # img_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/part'
    # fetch_max_width_height(src_dir)
    # src_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/DATA_CLEAN/damage_whole/img"
    # out_limit_dir = "/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-factory/DATA_CLEAN/damage_whole/limit"
    # move_limit_width_height(src_dir, out_limit_dir, 1920, 1920)
    # img_dir = '/home/ubuntu/zsf/dl/sample'
    # out_limit_dir = '/home/ubuntu/zsf/dl/sample-limit'
    # fetch_max_width_height(img_dir)
    # move_limit_width_height(img_dir,out_limit_dir,1024,768)
    # image_path = '/home/ubuntu/zsf/1.jpg'
    # img2gray(image_path)
    # gray2rgb_single(image_path)
    # show('/home/ubuntu/zsf/zhousf/213.JPG')
    # dir = '/home/ubuntu/zsf/zhousf/plate'
    # gray2rgb(dir)
    # enhance_brightness('/home/ubuntu/zsf/tf/1.jpg')
    # enhance_color('/home/ubuntu/zsf/tf/1.jpg')
    # enhance_contrast('/home/ubuntu/zsf/tf/11.jpg')
    # im = enhance_sharpness(img='/home/ubuntu/zsf/tf/11.jpg', show_img=False)
    # a = im.save('/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/1.jpg')
    # print(a)
    # src_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/whole/damage'
    # dst_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/segment/whole/damage_enhanced'
    # enhance(src_dir, dst_dir)
    # image = Image.fromarray(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
    # enhance_overwrite(image)

    root_dir = '/home/ubuntu/zsf/dl/plate/'
    src_dir = root_dir + 'data/'
    dst_dir = root_dir + 'data_generate_sample/'
    op = ImgOperator(src_dir, dst_dir)
    op.auto_deal()

    pass
