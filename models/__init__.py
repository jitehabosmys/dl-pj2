'''
Models implementation and training & evaluating functions
'''

from models.basic_cnn import BasicCNN
from models.resnet import ResNet18
from models.vgg import VGG_A, VGG_A_BatchNorm

# 更清晰地导出所有模型
__all__ = [
    'BasicCNN',      # 基础CNN
    'ResNet18',      # ResNet18
    'VGG_A',         # VGG系列
    'VGG_A_BatchNorm'
]