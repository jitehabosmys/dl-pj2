import torch
import os
import numpy as np

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # 单位：百万

def save_model(model, model_name, save_dir="results/models"):
    """保存模型"""
    # 如果指定的文件夹不存在，则创建它
    os.makedirs(save_dir, exist_ok=True)
    
    # 将模型保存在指定的文件夹路径下
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")

def load_model(model, model_name, save_dir="results/models"):
    """加载预训练模型"""
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
        return True
    else:
        print(f"Model file {model_path} not found")
        return False

def get_optimizer(model_or_params, opt_name='adam', lr=0.001, weight_decay=0):
    """根据名称获取优化器
    
    参数:
        model_or_params: 模型对象或参数列表
        opt_name: 优化器名称
        lr: 学习率
        weight_decay: 权重衰减系数
    """
    opt_name = opt_name.lower()
    
    # 判断输入是模型还是参数列表
    if isinstance(model_or_params, list):
        params = model_or_params
    else:
        params = model_or_params.parameters()
    
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def get_lr_scheduler(optimizer, scheduler_name='step', **kwargs):
    """获取学习率调度器"""
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'reduce':
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.1)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        return None 