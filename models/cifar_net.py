import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """基础CNN网络，包含全连接层、卷积层、池化层和激活函数"""
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))   # [32, 32, 32]
        x = self.pool(x)    # [32, 16, 16]
        x = F.relu(self.conv2(x))   # [64, 16, 16]
        x = self.pool(x)    # [64, 8, 8]
        x = F.relu(self.conv3(x))   # [128, 8, 8]
        x = self.pool(x)    # [128, 4, 4]
        x = torch.flatten(x, 1) # [128 * 4 * 4]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNWithBatchNorm(nn.Module):
    """带有BatchNorm的CNN网络"""
    def __init__(self):
        super(CNNWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 在激活之前归一化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNWithDropout(nn.Module):
    """带有Dropout的CNN网络"""
    def __init__(self, dropout_rate=0.25):
        super(CNNWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)     # 在池化之后随机丢弃
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    """残差块实现
    残差块由两个3x3卷积层组成，每个卷积后接BatchNorm和ReLU激活函数
    同时包含一个shortcut分支，实现残差连接（identity mapping）
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，可能会改变特征图大小和通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层，保持特征图大小和通道数不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 如果输入和输出维度不匹配（通道数或特征图大小不同），需要调整shortcut分支
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   
        out = self.bn2(self.conv2(out))         
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """带有残差连接的网络
    实现了简化版ResNet架构，针对CIFAR-10任务进行调整
    
    参数:
        num_blocks: 每个阶段的残差块数量
        num_classes: 分类类别数
    
    架构:
        1. 初始卷积层+BN
        2. 三个阶段的残差块，通道数分别为32,64,128（与BasicCNN一致）
        3. 全局平均池化
        4. 全连接分类层
    """
    def __init__(self, num_blocks=2, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32  # 从32开始，与其他模型保持一致
        
        # 初始卷积层，将输入从3通道转换为32通道（原来是64）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 三个阶段的残差块，通道数与BasicCNN保持一致
        # 第一阶段：保持特征图大小不变 (stride=1)
        self.layer1 = self._make_layer(32, num_blocks, stride=1)  # 输出: [32, 32, 32]
        # 第二阶段：将特征图大小减半 (stride=2)
        self.layer2 = self._make_layer(64, num_blocks, stride=2)  # 输出: [64, 16, 16]
        # 第三阶段：将特征图大小再次减半 (stride=2)
        self.layer3 = self._make_layer(128, num_blocks, stride=2) # 输出: [128, 8, 8]
        
        # 全局平均池化，将特征图转换为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层用于分类，输入维度调整为128
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        """创建包含多个残差块的层"""
        # 步长列表：第一个块可能使用stride>1进行降采样，后续块均使用stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        # 创建每个残差块
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            # 更新输入通道数为当前输出通道数，用于下一个块
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积和批量归一化
        out = F.relu(self.bn1(self.conv1(x)))  # [32, 32, 32]
        
        # 通过三个残差层
        out = self.layer1(out)                 # [32, 32, 32]
        out = self.layer2(out)                 # [64, 16, 16]
        out = self.layer3(out)                 # [128, 8, 8]
        
        # 全局平均池化
        out = self.avg_pool(out)               # [128, 1, 1]
        
        # 展平并通过全连接层
        out = torch.flatten(out, 1)            # [128]
        out = self.fc(out)                     # [10]
        
        return out 