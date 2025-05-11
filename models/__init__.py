'''
Models implementation and training & evaluating functions
'''

from . import vgg
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from models.vgg import VGG_A, VGG_A_BatchNorm
from models.pretrained_models import PretrainedResNet18, get_pretrained_resnet18