# -*- coding:utf-8 -*-
# Author:      zhousf
# Date:        2018-12-25
# File:        train_inception.py
# Description:  图片分类
import os
import warnings
import shutil
import numpy as np

MODEL_INCEPTION_V3 = 'inception_v3'
MODEL_INCEPTION_V4 = 'inception_v4'
MODEL_INCEPTION_RESNET_V2 = 'inception_resnet_v2'


class Inception(object):

    def __init__(self,
                 train_name='',
                 dataset_dir='',
                 gpu_with_train='0',
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
        self.checkpoint_exclude_scopes='%s/Logits,%s/AuxLogits'%(model_node_name,model_node_name)
        self.trainable_scopes='%s/Logits,%s/AuxLogits'%(model_node_name,model_node_name)
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.train_name = train_name
        self.gpu_with_train = gpu_with_train
        self.model_dir = os.getcwd() + '/models-master/research/slim'
        self.mymodels_dir = os.getcwd() + '/my_models'
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
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_with_train
        os.system(eval_command)

    def export(self):
        file_list = os.listdir(self.train_dir)
        check_file = []
        for i in range(0, len(file_list)):
            path = os.path.join(self.train_dir, file_list[i])
            if path.endswith(".index"):
                name, index = path.split("-")
                num, ext = index.split(".")
                check_file.append(int(num))
        point = max(check_file)
        if point > 0:
            checkpoint = self.trained_checkpoint + "-" + str(point)
            save_dir = self.save_model_dir + "/" + str(point)
        else:
            checkpoint = self.trained_checkpoint
            save_dir = self.save_model_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_pb = save_dir + '/inception.pb'
        shutil.copy(checkpoint + '.meta', save_dir)
        shutil.copy(checkpoint + '.index', save_dir)
        shutil.copy(checkpoint + '.data-00000-of-00001', save_dir)
        print(checkpoint)
        print(save_pb)
        export_command = 'python %s/export_inference_graph.py \
                         --model_name=%s \
                         --output_file=%s \
                         --dataset_name=%s \
                         --image_size=%d ' % (self.model_dir,
                                              self.model_name,
                                              save_pb,
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

    def vis_single_img(self, image_path, class_names_file=None, pb_model_path=None, show=True):
        if not os.path.exists(image_path):
            warnings.warn(image_path+'不存在')
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
        output_node_name = self.output_node_name + ':0'
        from slim import eval_single_image_inception
        num_top_predictions = 5
        result = eval_single_image_inception.infer(image_path,
                                                   pb_model_path,
                                                   class_names_file,
                                                   self.image_size,
                                                   num_top_predictions,
                                                   output_node_name)
        print (result)

    def vis(self):
        pass

    def show_eval(self, port=6007):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.eval_dir, port)
        os.system(tensor_board)

    def show_train(self, port=6006):
        tensor_board = 'tensorboard --logdir %s/ --port %d' % (self.train_dir, port)
        os.system(tensor_board)




class TrainFlowersV3(Inception):

    def __init__(self):
        model_name = MODEL_INCEPTION_V3
        train_name = 'flowers'
        dataset = 'flowers'
        dataset_dir = '../data/flowers'
        train_num = 100000
        batch_size = 32
        gpu_with_train = '0'
        Inception.__init__(self,
                           train_name=train_name,
                           dataset_dir=dataset_dir,
                           gpu_with_train=gpu_with_train,
                           model_name=model_name,
                           dataset=dataset,
                           train_num=train_num,
                           batch_size=batch_size)


class TrainFlowersV4(Inception):

    def __init__(self):
        model_name = MODEL_INCEPTION_V4
        train_name = 'flowers'
        dataset = 'flowers'
        dataset_dir = '../data/flowers'
        train_num = 500000
        batch_size = 32
        gpu_with_train = '1'
        Inception.__init__(self,
                           train_name=train_name,
                           dataset_dir=dataset_dir,
                           gpu_with_train=gpu_with_train,
                           model_name=model_name,
                           dataset=dataset,
                           train_num=train_num,
                           batch_size=batch_size)
