# 神经网络与深度学习课程 Project-2

本项目为神经网络与深度学习课程的第二个项目，主要实现了在CIFAR-10数据集上训练不同结构的神经网络模型，并探究了Batch Normalization对训练过程的影响。

## 项目概述

本项目包含两个主要部分：

1. **在CIFAR-10上训练神经网络模型**
   - 实现了多种网络结构，包括基础CNN、带BatchNorm的CNN、带Dropout的CNN以及ResNet
   - 提供了完整的训练、测试和比较功能
   - 支持多种优化器、学习率等超参数的实验

2. **探究Batch Normalization的效果与原理**
   - 比较带BN和不带BN的VGG-A网络性能
   - 分析BN对优化过程的影响（损失曲线、梯度等）

## 项目结构

```
.
├── data/                # 数据加载相关代码
├── models/              # 模型定义
├── scripts/             # 训练和测试脚本
│   ├── train.py         # 训练单个模型
│   ├── test.py          # 测试预训练模型
│   ├── compare.py       # 比较多个预训练模型
│   └── experiment.py    # 超参数实验
├── utils/               # 工具函数
│   ├── trainer.py       # 训练和评估函数
│   ├── model_utils.py   # 模型相关工具函数
│   └── visualization.py # 可视化相关函数
└── results/             # 实验结果（自动创建）
    ├── models/          # 保存的模型
    └── images/          # 生成的图像
```

## 安装与依赖

本项目主要依赖以下库：
- PyTorch
- torchvision
- matplotlib
- numpy

## 使用说明

### 1. 训练单个模型 (train.py)

使用此脚本训练单个模型并保存训练结果。

```bash
python scripts/train.py --model [MODEL_TYPE] [OPTIONS]
```

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 (`BasicCNN`, `CNNWithBatchNorm`, `CNNWithDropout`, `ResNet`) | 必填 |
| `--num_blocks` | ResNet的残差块数量 | 2 |
| `--dropout_rate` | Dropout的丢弃率 | 0.25 |
| `--epochs` | 训练轮数 | 6 |
| `--batch_size` | 批量大小 | 128 |
| `--validation_split` | 验证集比例，设为0则不使用验证集 | 0.1 |
| `--patience` | 早停耐心值，连续多少个epoch验证性能未提升则停止训练 | 5 |
| `--optimizer` | 优化器类型 (`adam`, `sgd`, `rmsprop`, `adamw`) | adam |
| `--lr` | 学习率 | 0.001 |
| `--weight_decay` | 权重衰减系数（L2正则化） | 0 |
| `--lr_scheduler` | 是否使用学习率调度器(ReduceLROnPlateau) | True |
| `--lr_patience` | 学习率调度器的耐心值，多少个epoch验证损失未改善则降低学习率 | 2 |
| `--lr_factor` | 学习率调度器的降低因子 | 0.1 |
| `--model_name` | 保存模型的自定义名称 | 模型类型名 |
| `--exp_tag` | 实验标签，会添加到保存文件名中 | 空 |
| `--output_dir` | 输出目录 | results |
| `--download` | 是否下载CIFAR-10数据集 | True |
| `--seed` | 随机种子 | 42 |
| `--cuda` | 是否使用CUDA | True |

**示例：**

```bash
# 训练基础CNN模型
python scripts/train.py --model BasicCNN --epochs 10 --lr 0.001

# 训练带BatchNorm的CNN并自定义模型名称
python scripts/train.py --model CNNWithBatchNorm --model_name cnn_bn_custom --lr 0.0005
```

### 2. 测试预训练模型 (test.py)

使用此脚本测试已训练的模型在测试集上的性能。

```bash
python scripts/test.py --model [MODEL_TYPE] [OPTIONS]
```

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 (`BasicCNN`, `CNNWithBatchNorm`, `CNNWithDropout`, `ResNet`) | 必填 |
| `--num_blocks` | ResNet的残差块数量 | 2 |
| `--dropout_rate` | Dropout的丢弃率 | 0.25 |
| `--batch_size` | 批量大小 | 128 |
| `--model_name` | 要加载的模型文件名 | 模型类型名 |
| `--model_dir` | 模型保存目录 | results/models |
| `--cuda` | 是否使用CUDA | True |
| `--verbose` | 是否输出详细信息 | False |

**示例：**

```bash
# 测试基础CNN模型
python scripts/test.py --model BasicCNN

# 测试自定义名称的模型
python scripts/test.py --model CNNWithBatchNorm --model_name cnn_bn_custom --verbose
```

### 3. 比较多个预训练模型 (compare.py)

使用此脚本比较多个预训练模型的性能并生成可视化结果。

```bash
python scripts/compare.py --models [MODEL_TYPES] [OPTIONS]
```

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 要比较的模型类型，可指定多个或使用 `all` | ['all'] |
| `--batch_size` | 批量大小 | 128 |
| `--comparison_name` | 比较实验的自定义名称 | model_comparison |
| `--model_names` | 要加载的各模型的自定义文件名 | None |
| `--model_dir` | 模型保存目录 | results/models |
| `--output_dir` | 输出目录 | results |
| `--cuda` | 是否使用CUDA | True |
| `--save_results` | 是否保存比较结果到JSON | False |

**示例：**

```bash
# 比较所有模型
python scripts/compare.py --models all --comparison_name experiment1

# 比较特定模型
python scripts/compare.py --models BasicCNN CNNWithBatchNorm --save_results
```

### 4. 超参数实验 (experiment.py)

使用此脚本进行超参数组合实验，测试不同优化器、学习率等参数的影响。

```bash
python scripts/experiment.py --model [MODEL_TYPE] [OPTIONS]
```

**主要参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型 | 必填 |
| `--exp_name` | 实验名称 | hyperparameter_experiment |
| `--optimizers` | 要测试的优化器，可多选或 `all` | ['adam'] |
| `--learning_rates` | 要测试的学习率列表 | [0.001] |
| `--weight_decays` | 要测试的权重衰减系数列表 | [0] |
| `--schedulers` | 要测试的学习率调度器类型，可多选 | ['none'] |
| `--epochs` | 训练轮数 | 6 |
| `--batch_size` | 批量大小 | 128 |
| `--output_dir` | 输出目录 | results/experiments |
| `--save_all_models` | 是否保存所有模型 | False |

**示例：**

```bash
# 测试不同优化器
python scripts/experiment.py --model BasicCNN --optimizers all --learning_rates 0.001 0.0001

# 测试不同学习率调度器
python scripts/experiment.py --model CNNWithBatchNorm --schedulers all --exp_name bn_scheduler_test
```

## 实验示例

以下是一些完整的实验示例流程：

**1. 训练和测试单个模型：**

```bash
# 训练模型
python scripts/train.py --model CNNWithBatchNorm --epochs 10 --optimizer adam --lr 0.001

# 测试训练好的模型
python scripts/test.py --model CNNWithBatchNorm --verbose
```

**2. 对比多个模型结构：**

```bash
# 训练不同模型
python scripts/train.py --model BasicCNN --epochs 10
python scripts/train.py --model CNNWithBatchNorm --epochs 10
python scripts/train.py --model CNNWithDropout --epochs 10

# 比较模型并生成可视化结果
python scripts/compare.py --models BasicCNN CNNWithBatchNorm CNNWithDropout --comparison_name structure_comparison --save_results
```

**3. 探究BatchNorm的影响：**

```bash
# 训练带BatchNorm和不带BatchNorm的模型
python scripts/train.py --model BasicCNN --epochs 15 --model_name CNN_without_BN
python scripts/train.py --model CNNWithBatchNorm --epochs 15 --model_name CNN_with_BN

# 比较两种模型
python scripts/compare.py --models BasicCNN CNNWithBatchNorm --model_names CNN_without_BN CNN_with_BN --comparison_name bn_effect
```
