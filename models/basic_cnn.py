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