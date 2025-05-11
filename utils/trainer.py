import torch
from tqdm import tqdm
import numpy as np

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    """设置所有可能的随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    """训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epochs: 训练轮数
        scheduler: 学习率调度器（可选）
    
    返回:
        train_losses: 每个epoch的训练损失列表
        train_accs: 每个epoch的训练准确率列表
    """
    model.train()
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}')
        
        # 如果提供了学习率调度器，更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)  # 对于ReduceLROnPlateau，需要提供指标
            else:
                scheduler.step()  # 对于其他调度器，直接step
    
    return train_losses, train_accs

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Evaluating') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{test_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy 