# Contrastive

**项目取材自 2024 年 ”泰迪杯“ 数据挖掘挑战赛 B 题，基于共享特征空间对比学习的跨模态图文互检模型**

## 项目简介

本项目的原理与 OpenAI 推出的 [CLIP](https://github.com/openai/CLIP) 类似，目前主要作学习和研究用途。

### 模型结构

模型视觉侧采用 ResNet50 模型作为图像特征提取器，文本侧采用 RoBERTa 模型作为文本特征提取器，通过一个全连接层将图像特征向量映射到和文本特征向量相同的维度。最后将提取得到的图像特征向量和文本特征向量作归一化后输出。

### 训练策略

由于图像特征和文本特征均进行了归一化处理，因此可以将两个向量的内积作为图像特征和文本特征的相似度，其值处于 [-1, 1] 的区间内，值越大说明两个特征越相似，反之则越不相似。

训练集由若干的图像以及对应的文本描述构成，图像和对应的文本描述构成一个图文对。在每一批的训练样本图文对中，经由模型提取特征后，两两之间求内积，可以得到一个相似度矩阵。

对于这样的一个相似度矩阵，对角线上的值为原本能够匹配的图文相似度，因此我们希望对角线上的值越大越好，其余的值越小越好。对于相似度矩阵的每一行，可以视为由图像到文本的分类问题；对于相似度矩阵的每一列，可以视为由文本到到图像的分类问题，因此分别对于相似度矩阵及其转置矩阵，以索引序列为标签，计算交叉熵损失并求和作为模型训练的优化目标。

<u>*注：由此看来，在条件允许的情况下，训练的 batch size 越大，模型能够 “见识” 到的不同特征模式也就越多，对比学习的效果也就越好。*</u>

## 性能评估

模型的训练集包括 [COCO2017](https://cocodataset.org/#download) 数据集中训练集部分和 [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) 数据集中的全部图像及文本描述，共计约 15 万张图像，每张图像有 5 条对应的文本描述，共计约 75 万图文对。

模型的验证集包括 COCO2017 数据集中的验证集部分，共计 5000 张图像，每张图像选取其中 1 条对应的文本描述，共计 5000 图文对。

对于训练好的模型，采用 R @ k 召回率在验证集上对 5000 图文对作全量召回评估，即模型需要根据给定的图像，在 5000 条文本中找出最能与之相符的前 k 条文本，反之亦然。

截至目前，训练得到的最优模型评估结果如下：

|           | R @ 1   | R @ 5   | R @ 10  |
|:---------:| ------- | ------- | ------- |
| **图到文检索** | 0.28180 | 0.59600 | 0.72880 |
| **文到图检索** | 0.29020 | 0.58460 | 0.71260 |

## 使用说明

### 环境搭建

首先需要安装本项目依赖的各种库和工具包。

```bash
pip install -r requirements.txt
```

### 数据集格式

本项目的训练数据集格式如下，分为训练集和验证集，所有的图像文件需要放入相应的 images/ 目录下（最好保证图像宽高一致）。

```shell
datasets/
├── train/
│   ├── images/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   └── captions.csv
└── valid/
    ├── images/
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   └── ...
    └── captions.csv
```

其中 captions.csv 中的每一行记录了数据集中的一个图文对，列 'filename' 为图像文件名，需要与 images/ 目录下的图像文件名一致，列 'caption' 为该图像对应的文本描述，其格式如下：

```csv
filename,caption
000000000139.jpg,A woman stands in the dining area at the table.
000000000285.jpg,A big burly grizzly bear is show with grass in the background.
```

### 模型准备

由于众所周知的原因，您需要想办法手动从 [Hugging Face](https://huggingface.co/) 上下载 RoBERTa 模型的相关文件和预训练权重到 <u>weights/develop/roberta/</u> 目录下，但 ResNet50 模型的预训练权重能够自动下载，随后运行 prepare.py 文件，这将会加载 ResNet50 和 RoBERTa 模型的模型结构和预训练权重并进行整合，作为一个联合模型保存到本地用于下一步的训练工作。对应的配置文件为 <u>configs/model.yaml</u>，默认配置及其含义如下：

```yaml
pretrained: true                            # 是否加载预训练参数
save-path: "weights/develop/pretrain.pt"    # 模型输出路径
```

### 模型训练

准备好数据集和初始模型后，运行 train.py 开始训练，默认仅支持使用 GPU 进行训练，因为这种规模的模型在 CPU 上几乎不可能完成训练。默认的训练配置文件是 <u>configs/train.yaml</u>，默认配置及其含义如下：

```yaml
epochs: 200                    # 训练迭代轮数
learning-rate: 0.000005        # 学习率
batch-size: 192                # 批大小
num-workers: 8                 # 数据加载子进程数
use-amp: true                  # 是否启动 AMP（自动混合精度），开启有助于减少显存占用并加速训练
use-augment: true              # 是否启用图像数据增强
temperature: 0.07              # 损失函数温度参数，具体作用有待验证

load-path: "weights/develop/pretrain.pt"    # 初始模型加载路径
best-path: "weights/develop/best.pt"        # 当前验证集最优模型保存路径
last-path: "weights/develop/last.pt"        # 最后一次训练模型保存路径
```

### 模型评估

模型训练完成后，运行 eval.py 进行评估。这将会分别计算模型在验证集上的 R @ k 图到文召回率和文到图召回率。默认的配置文件为 <u>configs/eval.yaml</u>，默认配置及其含义如下：

```yaml
batch-size: 256                            # 批大小
num-workers: 8                             # 数据加载子进程数
use-amp: true                              # 是否启动 AMP（自动混合精度）
model-path: "weights/develop/best.pt"      # 待评估模型加载路径
```

## 写在最后

这个项目源自 2024 年 ”泰迪杯“ 数据挖掘挑战赛 B 题《基于多模态特征融合的图像文本检索》，当时参赛时用的是 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)，但没能训练成功，最后也只勉强获得了省三。后来想要自己建模尝试解决这个问题，一开始我尝试了许多不同的方案，用了各种各样的模型和数据集，都没能取得好的结果。在 ChatGPT 的帮助下，改着改着就和 CLIP 越来越相似了。

作为一名学生，时间和经费都有限，难以在大规模的数据集上开展训练，也没有条件作更细致的超参数调优，最后得到的模型效果也只能说是一般。如果对这个课题感兴趣的话可以尝试着在更大规模的数据集上进行训练，尝试不同的超参数组合，或是换用更加复杂的模型结构，看看是否能够得到更好的模型。您可以在我训练结果的基础上继续训练，如有需要可以在本项目 Issues 中提出并附上您的电子邮箱地址，我会将我当前得到的模型权重发送给您。

在训练的过程中可能会遇到随着训练的进行，模型的准确率不升反降的情况。不必担心，这是由于数据集规模不足所致，通用的跨模态图文互检模型需要处理海量的特征模式，如果数据集的规模太小，同类型的样本对数量太少，模型极有可能将数据中的噪声当作特征进行学习，从而导致在验证集上的表现不佳。其实这个问题从我参赛时开始就一直困扰着我，无论是自己建模还是使用开源模型均是如此，反复尝试各种模型结构也都无济于事，这也是导致我在比赛中没能成功训练模型的原因。最后我将数据集的规模由原来的 4 万图文对增加到目前的 75 万图文对，情况才有所缓解。

这个项目仍在建设中，未来计划增加模型的推理部署，图文检索以及零样本图像分类的后端 Web 服务和前端交互界面，使之构成一个完整的跨模态图文互检系统。也会考虑尝试使用中文数据集训练中文版本的模型。
