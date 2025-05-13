'''
Models implementation and training & evaluating functions
'''

from models.basic_cnn import BasicCNN
from models.resnet import ResNet18
from models.vgg import VGG_A, VGG_A_BatchNorm
from models.preact_resnet18 import PreActResNet18
from models.pretrained_models import PretrainedResNet18, get_pretrained_resnet18

# 更清晰地导出所有模型
__all__ = [
    'BasicCNN',      # 基础CNN
    'ResNet18',      # ResNet18
    'VGG_A',         # VGG系列
    'VGG_A_BatchNorm',
    'PreActResNet18', # 预激活ResNet18
    'PretrainedResNet18', # 预训练ResNet18
    'get_pretrained_resnet18' # 创建预训练ResNet18的工厂函数
]