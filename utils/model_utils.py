import torch
import os
import numpy as np

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # 单位：百万

def save_model(model, model_name, optimizer=None, scheduler=None, epoch=None, best_acc=None, best_val_loss=None, save_dir="results/models"):
    """保存模型和训练状态
    
    参数:
        model: 模型对象
        model_name: 模型名称
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前训练轮次
        best_acc: 最佳准确率
        best_val_loss: 最佳验证损失
        save_dir: 保存目录
    """
    # 如果指定的文件夹不存在，则创建它
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备保存的状态字典
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'acc': best_acc,
        'best_val_loss': best_val_loss
    }
    
    # 如果提供了优化器，保存其状态
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    # 如果提供了学习率调度器，保存其状态
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    
    # 将模型保存在指定的文件夹路径下
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(state, model_path)
    
    print(f"模型和训练状态保存到 {model_path}")

def load_model(model, model_name, optimizer=None, scheduler=None, save_dir="results/models"):
    """加载预训练模型和训练状态
    
    参数:
        model: 模型对象
        model_name: 模型名称
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        save_dir: 保存目录
        
    返回:
        success: 是否成功加载
        state: 加载的状态字典
    """
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        
        # 修复键名：去除 `module.` 前缀（兼容 DataParallel 保存的模型）
        def fix_state_dict(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")  # 关键修复：去掉前缀
                new_state_dict[name] = v
            return new_state_dict
        
        # 加载模型权重
        if 'net' in checkpoint:
            model.load_state_dict(fix_state_dict(checkpoint['net']))  # 应用修复
        else:
            # 处理旧格式的保存文件（同样需要修复键名）
            model.load_state_dict(fix_state_dict(checkpoint))         # 应用修复
        
        # 加载优化器状态（如果提供）
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载学习率调度器状态（如果提供）
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"模型和训练状态从 {model_path} 加载完成")
        return True, checkpoint
    else:
        print(f"模型文件 {model_path} 未找到")
        return False, None

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