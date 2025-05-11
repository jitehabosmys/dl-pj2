import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedResNet18(nn.Module):
    """使用预训练的ResNet18模型，并适配CIFAR-10数据集
    
    参数:
        num_classes: 分类类别数，默认为10（CIFAR-10）
        pretrained: 是否加载预训练权重
        first_conv_adapt: 是否调整第一个卷积层以适应小尺寸图像
    """
    def __init__(self, num_classes=10, pretrained=True, first_conv_adapt=True):
        super(PretrainedResNet18, self).__init__()
        
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(pretrained=pretrained)
        
        # 如果需要调整第一个卷积层以适应CIFAR-10的32x32图像
        if first_conv_adapt:
            # 原始ResNet的第一个卷积层: 7x7 kernel, stride 2, padding 3
            # 我们替换为更适合小图像的3x3 kernel, stride 1, padding 1
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # 也可以移除或调整 maxpool 层，因为在小图像上效果不理想
            self.model.maxpool = nn.Identity()
        
        # 替换最后的全连接层以适应CIFAR-10的类别数
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self, unfreeze_last_n_layers=0):
        """冻结模型主干，只训练最后几个层
        
        参数:
            unfreeze_last_n_layers: 保持解冻的最后几个层的数量
        """
        # 首先冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 解冻最后的全连接层
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        # 可选：解冻最后几个层
        if unfreeze_last_n_layers > 0:
            layers_to_unfreeze = []
            # 解冻layer4的最后n个块
            if unfreeze_last_n_layers >= 1:
                layers_to_unfreeze.append(self.model.layer4[-1])
            if unfreeze_last_n_layers >= 2:
                layers_to_unfreeze.append(self.model.layer4[-2])
            if unfreeze_last_n_layers >= 3:
                layers_to_unfreeze.append(self.model.layer3[-1])
            
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
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
            - "last_layer": 只训练最后的全连接层
            - "partial": 训练最后的全连接层和最后2个残差块
    
    返回:
        配置好的PretrainedResNet18模型
    """
    model = PretrainedResNet18(num_classes=num_classes, pretrained=pretrained)
    
    if finetune_mode == "last_layer":
        model.freeze_backbone(unfreeze_last_n_layers=0)
    elif finetune_mode == "partial":
        model.freeze_backbone(unfreeze_last_n_layers=2)
    
    return model 