import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class PretrainedResNet18(nn.Module):
    """使用预训练的ResNet18模型，并适配CIFAR-10数据集
    
    参数:
        num_classes: 分类类别数，默认为10（CIFAR-10）
        pretrained: 是否加载预训练权重
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(PretrainedResNet18, self).__init__()
        
        # 加载预训练的ResNet18模型
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = models.resnet18(weights=weights)
        
        # 调整第一个卷积层以适应CIFAR-10的32x32图像
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除maxpool层，使用Identity代替
        self.model.maxpool = nn.Identity()
        
        # 替换最后的全连接层以适应CIFAR-10的类别数
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """冻结模型主干，只训练最后一层"""
        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 解冻最后的全连接层
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """获取当前可训练的参数"""
        return [p for p in self.parameters() if p.requires_grad]

def get_pretrained_resnet18(num_classes=10, pretrained=True, finetune_mode="full"):
    """创建预训练ResNet18模型的工厂函数
    
    参数:
        num_classes: 分类类别数
        pretrained: 是否加载预训练权重
        finetune_mode: 微调模式
            - "full": 训练整个网络
            - "last": 只训练最后的全连接层
    
    返回:
        配置好的PretrainedResNet18模型
    """
    model = PretrainedResNet18(num_classes=num_classes, pretrained=pretrained)
    
    if finetune_mode == "last":
        model.freeze_backbone()
    
    return model 