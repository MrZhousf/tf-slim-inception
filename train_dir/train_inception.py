# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        train_inception.py
# Description:  图片分类
import os
import warnings
import shutil
import time
import numpy as np
from prettytable import PrettyTable
from train_dir.infer_inception import Prediction

MODEL_INCEPTION_V3 = 'inception_v3'
MODEL_INCEPTION_V4 = 'inception_v4'
MODEL_INCEPTION_RESNET_V2 = 'inception_resnet_v2'


class Inception(object):

    def __init__(self,
                 train_name='',
                 dataset_dir='',
                 gpu_with_train='0',
                 gpu_with_eval='0',
                 model_name=MODEL_INCEPTION_V3,
                 dataset='flowers',
                 train_num=10000000,
                 batch_size=32,
                 learning_rate=0.01,
                 image_size=299):
        self.model_name = model_name
        model_node_name = ''
        if self.model_name == MODEL_INCEPTION_V3:
            model_node_name = 'InceptionV3'
            self.output_node_name = 'InceptionV3/Predictions/Reshape_1'
        elif self.model_name == MODEL_INCEPTION_RESNET_V2:
            model_node_name = 'InceptionResnetV2'
            self.output_node_name = 'InceptionResnetV2/Logits/Predictions'
        elif self.model_name == MODEL_INCEPTION_V4:
            model_node_name = 'InceptionV4'
            self.output_node_name = 'InceptionV4/Logits/Predictions'
        else:
            warnings.warn('不支持模型：'+model_name)
        self.classify = None
        self.pb_model_path = None
        self.checkpoint_exclude_scopes='%s/Logits,%s/AuxLogits'%(model_node_name,model_node_name)
        self.trainable_scopes='%s/Logits,%s/AuxLogits'%(model_node_name,model_node_name)
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.train_name = train_name
        self.gpu_with_train = gpu_with_train
        self.gpu_with_eval = gpu_with_eval
        self.model_dir = os.path.dirname(os.getcwd()) + '/model/research/slim'
        self.mymodels_dir = os.path.dirname(os.getcwd()) + '/train_dir'
        self.initial_checkpoint = self.mymodels_dir + '/' + model_name + '/' + model_name + '.ckpt'
        self.dataset = dataset
        self.train_num = train_num
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.class_names_file = dataset_dir + '/labels.txt'
        train_model = self.mymodels_dir + "/" + self.model_name + "/" + self.train_name
        class_names_file = train_model + '/labels.txt'
        if not os.path.exists(class_names_file):
            if not os.path.exists(self.class_names_file):
                warnings.warn(self.class_names_file + '不存在')
            if not os.path.exists(train_model):
                os.makedirs(train_model)
            shutil.copy(self.class_names_file, train_model)
        self.class_names_file = class_names_file
        with open(self.class_names_file, 'r') as load_f:
            file_context = load_f.read().splitlines()
            class_names = np.asarray(file_context)
            num_classes = len(class_names)
            self.num_classes = num_classes
            self.batch_size = batch_size
            # 日志目录
            self.log_dir = train_model + "/log"
            # 可视化目录
            self.vis_dir = train_model + "/vis"
            # 训练目录
            self.train_dir = train_model + "/train"
            # 评估目录
            self.eval_dir = train_model + "/eval"
            # 保存模型目录
            self.save_model_dir = train_model + "/export"
            # 保存训练模型文件
            self.trained_checkpoint = self.train_dir + "/model.ckpt"
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)
            if not os.path.exists(self.eval_dir):
                os.makedirs(self.eval_dir)
            if not os.path.exists(self.save_model_dir):
                os.makedirs(self.save_model_dir)
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

    def train(self):
        train_command = 'python %s/train_image_classifier.py \
            --train_dir=%s \
            --dataset_name=%s \
            --dataset_split_name=train \
            --dataset_dir=%s \
            --model_name=%s \
            --checkpoint_path=%s \
            --checkpoint_exclude_scopes=%s \
            --trainable_scopes=%s \
            --max_number_of_steps=%d \
            --batch_size=%d \
            --learning_rate=%f \
            --learning_rate_decay_type=fixed \
            --image_size=%d \
            --log_every_n_steps=10 \
            --optimizer=rmsprop \
            --weight_decay=0.00004 ' % (self.model_dir,
                                        self.train_dir,
                                        self.dataset,
                                        self.dataset_dir,
                                        self.model_name,
                                        self.initial_checkpoint,
                                        self.checkpoint_exclude_scopes,
                                        self.trainable_scopes,
                                        self.train_num,
                                        self.batch_size,
                                        self.learning_rate,
                                        self.image_size)
        if self.gpu_with_train == '':
            train_command += '--clone_on_cpu=True'
        else:
            train_command += '--clone_on_cpu=False'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_train
        os.system(train_command)

    def eval(self):
        eval_command = 'python %s/eval_image_classifier.py \
                       --checkpoint_path=%s \
                       --dataset_name=%s \
                       --eval_dir=%s \
                       --dataset_split_name=validation \
                       --dataset_dir=%s \
                       --model_name=%s \
                       --image_size=%d \
                       --batch_size=%d' % (self.model_dir,
                                           self.train_dir,
                                           self.dataset,
                                           self.eval_dir,
                                           self.dataset_dir,
                                           self.model_name,
                                           self.image_size,
                                           self.batch_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_eval
        os.system(eval_command)

    @staticmethod
    def fetch_max_ckpt(train_dir):
        """
        获取train_dir中最大ckpt
        :param train_dir:
        :return:
        """
        file_list = os.listdir(train_dir)
        check_file = []
        for i in range(0, len(file_list)):
            path = os.path.join(train_dir, file_list[i])
            if path.endswith(".index"):
                p = os.path.basename(path)
                name, index = p.split("-")
                num, ext = index.split(".")
                check_file.append(int(num))
        point = max(check_file)
        return point

    def export(self):
        point = self.fetch_max_ckpt(self.train_dir)
        if point > 0:
            checkpoint = self.trained_checkpoint + "-" + str(point)
            save_dir = self.save_model_dir + "/" + str(point)
        else:
            checkpoint = self.trained_checkpoint
            save_dir = self.save_model_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_pb = save_dir + '/saved_model.pb'
        shutil.copy(checkpoint + '.meta', save_dir)
        shutil.copy(checkpoint + '.index', save_dir)
        shutil.copy(checkpoint + '.data-00000-of-00001', save_dir)
        print(checkpoint)
        export_command = 'python %s/export_inference_graph.py \
                         --model_name=%s \
                         --output_file=%s \
                         --dataset_dir=%s \
                         --dataset_name=%s \
                         --image_size=%d ' % (self.model_dir,
                                              self.model_name,
                                              save_pb,
                                              self.dataset_dir,
                                              self.dataset,
                                              self.image_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.system(export_command)
        from tensorflow.python.tools import freeze_graph
        input_graph = save_pb
        input_checkpoint = checkpoint
        output_graph = save_dir + '/frozen_inference_graph.pb'
        freeze_command = 'python -u %s \
                         --input_graph=%s \
                         --input_checkpoint=%s \
                         --output_graph=%s \
                         --input_binary=True \
                         --output_node_name=%s ' % (os.path.abspath(freeze_graph.__file__),
                                                    input_graph,
                                                    input_checkpoint,
                                                    output_graph,
                         self.output_node_name)
        os.system(freeze_command)

    def vis_single_img(self, image_path, class_names_file=None, pb_model_path=None, num_top_predictions=5):
        if self.pb_model_path is None:
            if not os.path.exists(image_path):
                warnings.warn(image_path + '不存在')
                return
            if class_names_file is None:
                class_names_file = self.class_names_file
            if pb_model_path is None:
                file_list = os.listdir(self.save_model_dir)
                check_file = []
                for i in range(0, len(file_list)):
                    if os.path.isdir(os.path.join(self.save_model_dir, file_list[i])):
                        check_file.append(int(file_list[i]))
                if len(check_file) == 0:
                    warnings.warn('frozen_inference_graph.pb不存在')
                    return
                max_num = str(max(check_file))
                pb_model_path = self.save_model_dir + '/' + max_num + '/frozen_inference_graph.pb'
                if not os.path.exists(pb_model_path):
                    warnings.warn(pb_model_path + '不存在')
                    return
                self.pb_model_path = pb_model_path
        if self.classify is None:
            print(pb_model_path)
            self.classify = Prediction(pb_file=pb_model_path,
                                       label_file=class_names_file,
                                       gpu_assigned=self.gpu_with_train,
                                       num_top_predictions=num_top_predictions,
                                       model_type=Prediction.INCEPTION_V3)
        result = self.classify.infer(image_file=image_path)
        return result

    def vis(self):
        pass

    def show_eval(self, port=6017):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.eval_dir, port)
        os.system(tensor_board)

    def show_train(self, port=6016):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.train_dir, port)
        os.system(tensor_board)

    def show_accuracy(self, img_dir, pb_index=None):
        if not os.path.exists(img_dir):
            return warnings.warn('目录不存在：{0}'.format(img_dir))
        file_list = os.listdir(self.save_model_dir)
        check_file = []
        for i in range(0, len(file_list)):
            if os.path.isdir(os.path.join(self.save_model_dir, file_list[i])):
                check_file.append(int(file_list[i]))
        if len(check_file) == 0:
            warnings.warn('frozen_inference_graph.pb不存在')
            return
        max_num = max(check_file) if pb_index is None else pb_index
        pb_dir = self.save_model_dir + '/' + str(max_num)
        pb_model_path = pb_dir + '/frozen_inference_graph.pb'
        if not os.path.exists(pb_model_path):
            warnings.warn(pb_model_path + '不存在')
            return
        print(pb_model_path)
        self.classify = Prediction(pb_file=pb_model_path,
                                   label_file=self.class_names_file,
                                   num_top_predictions=1,
                                   gpu_assigned=self.gpu_with_train,
                                   model_type=self.output_node_name + ":0")
        total = 0
        true_num = 0
        res = {}
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                start = time.time()
                current_file = os.path.join(root, file)
                class_name = os.path.basename(os.path.dirname(current_file))
                result = self.classify.infer(current_file)
                if result[0][0] == class_name:
                    true_num += 1
                    if class_name in res:
                        (cls_count, cls_total) = res.get(class_name)
                        cls_count += 1
                        cls_total += 1
                        res[class_name] = (cls_count, cls_total)
                    else:
                        res[class_name] = (1, 1)
                else:
                    if class_name in res:
                        (cls_count, cls_total) = res.get(class_name)
                        cls_total += 1
                        res[class_name] = (cls_count, cls_total)
                    else:
                        res[class_name] = (0, 1)
                total += 1
                print('{0} Time cost：{1}s {2}'.format(total, time.time() - start, result))
        table = PrettyTable(["class_name", "correct total", "total of all", "accuracy"])
        for cls_des in res:
            (cls_count, cls_total) = res.get(cls_des)
            cls_total = 1 if cls_total == 0 else cls_total
            cls_rate = (cls_count / cls_total) * 100
            p_rate = "%.2f" % cls_rate + "%"
            table.add_row([cls_des, cls_count, cls_total, str(p_rate) + "%"])
        table.align["class_name"] = "l"
        total = 1 if total == 0 else total
        rate = (true_num / total) * 100
        rate = "%.2f" % rate + "%"
        print(table)
        print("Accuracy={0}/{1}={2}".format(true_num, total, rate))
        with open(os.path.join(pb_dir, "accuracy.txt"), 'w') as f:
            f.write("Accuracy={0}/{1}={2}\n".format(true_num, total, rate))
            f.write('%s\n' % str(table))


class TrainFlowersV3(Inception):

    def __init__(self):
        model_name = MODEL_INCEPTION_V3
        train_name = 'flowers'
        dataset = 'flowers'
        dataset_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/flowers'
        train_num = 1000
        batch_size = 32
        gpu_with_train = '1'
        gpu_with_eval = '1'
        Inception.__init__(self,
                           train_name=train_name,
                           dataset_dir=dataset_dir,
                           gpu_with_train=gpu_with_train,
                           gpu_with_eval=gpu_with_eval,
                           model_name=model_name,
                           dataset=dataset,
                           learning_rate=0.01,
                           train_num=train_num,
                           batch_size=batch_size)


class TrainFlowersV4(Inception):

    def __init__(self):
        model_name = MODEL_INCEPTION_V4
        train_name = 'flowers'
        dataset = 'flowers'
        dataset_dir = '/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/flowers'
        train_num = 1000
        batch_size = 32
        gpu_with_train = '0'
        gpu_with_eval = '0'
        Inception.__init__(self,
                           train_name=train_name,
                           dataset_dir=dataset_dir,
                           gpu_with_train=gpu_with_train,
                           gpu_with_eval=gpu_with_eval,
                           model_name=model_name,
                           dataset=dataset,
                           learning_rate=0.01,
                           train_num=train_num,
                           batch_size=batch_size)


if __name__ == '__main__':
    model = TrainFlowersV4()
    model.train()
    # model.eval()
    # model.show_train()
    # model.show_eval()
    # model.export()
    # print(model.vis_single_img("/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/test/sorter/id card/0a565f44c5c5d45cbca4b2d6702af268.jpg"))
    # model.show_accuracy("/media/ubuntu/b8f80802-d95a-41c3-b157-6f4e34967425/data-zhousf/sorter/test_images")

