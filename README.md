## tf-slim-inception
tensorflow-slim下的inception_v3、inception_v4、inception_resnet_v2分类模型的数据制作、训练、评估、导出模型、测试。
训练比较请参考：[InceptionV3、InceptionV4图像分类训练与比较](https://blog.csdn.net/zsf442553199/article/details/85683335)


## 相关截图
### 项目结构
![](https://github.com/MrZhousf/tf-slim-inception/blob/master/pic/10.png?raw=true)

### 路径配置
将path.sh中的路径修改成自己的路径即可
```
#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/Users/zhousf/tensorflow/zhousf/tf-slim-inception/model
export PYTHONPATH=$PYTHONPATH:/Users/zhousf/tensorflow/zhousf/tf-slim-inception/model/research
export PYTHONPATH=$PYTHONPATH:/Users/zhousf/tensorflow/zhousf/tf-slim-inception/model/research/slim
export PYTHONPATH=$PYTHONPATH:/Users/zhousf/tensorflow/zhousf/tf-slim-inception/model/research/slim/datasets
export PYTHONPATH=$PYTHONPATH:/Users/zhousf/tensorflow/zhousf/tf-slim-inception
```

### 数据制作
generate_tf_record_file.py
![](https://github.com/MrZhousf/tf-slim-inception/blob/master/pic/11.png?raw=true)

自动生成训练集与评估集描述文件：train_eval_num.txt
```html
{"train": 3306, "validation": 364, "classes_num": 5}

+------------+-----------+----------+-------+
| class_name | train_num | eval_num | total |
+------------+-----------+----------+-------+
| tulips     |    720    |    79    |  799  |
| roses      |    577    |    64    |  641  |
| sunflowers |    630    |    69    |  699  |
| daisy      |    570    |    63    |  633  |
| dandelion  |    809    |    89    |  898  |
+------------+-----------+----------+-------+
```


### 训练
* train.py/train.sh
* 终端中运行：source train.sh 即可

### 评估
* eval.py/eval.sh
* 终端中运行：source eval.sh 即可

### 可视化
* show_train.py 训练
* show_eval.py 评估

### 导出模型
* export.py/export.sh
* 终端中运行：source export.sh 即可

### 测试
eval_single_img.py

### 模型配置文件
* train_inception.py 配置训练的参数(网络模型选择，训练次数，batch_size、指定GPU等)
* config.py 指定训练的业务

