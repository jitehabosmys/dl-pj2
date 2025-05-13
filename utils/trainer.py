import torch
from tqdm import tqdm
import numpy as np
import wandb  # 导入wandb

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    """设置所有可能的随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_gradient_norm(model):
    """计算模型参数的梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train(model, train_loader, criterion, optimizer, device, epochs=10, scheduler=None, validation_loader=None, patience=5, best_val_loss=float('inf'), use_wandb=False):
    """训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epochs: 训练轮数
        scheduler: 学习率调度器（可选）
        validation_loader: 验证数据加载器（可选）
        patience: 早停耐心值，连续多少个epoch验证性能未提升则停止训练
        best_val_loss: 初始最佳验证损失（用于恢复训练）
        use_wandb: 是否使用wandb记录训练过程
    
    返回:
        train_losses: 每个epoch的训练损失列表
        train_accs: 每个epoch的训练准确率列表
        val_losses: 每个epoch的验证损失列表（如果使用验证集）
        val_accs: 每个epoch的验证准确率列表（如果使用验证集）
        best_model_state: 最佳模型状态
    """
    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 早停相关变量
    best_model_state = model.state_dict().copy()
    early_stopping_counter = 0
    
    # 记录梯度范数
    grad_norms = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch_grad_norms = []
        
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
                
                # 计算并记录梯度范数
                grad_norm = get_gradient_norm(model)
                batch_grad_norms.append(grad_norm)
                
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'grad_norm': f'{grad_norm:.4f}'
                })
        
        # 计算每个epoch的平均梯度范数
        epoch_grad_norm = sum(batch_grad_norms) / len(batch_grad_norms)
        grad_norms.append(epoch_grad_norm)
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # wandb记录当前epoch的训练指标
        if use_wandb:
            try:
                wandb_log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "gradient_norm": epoch_grad_norm
                }
            except Exception as e:
                print(f"准备wandb训练指标时出错: {e}")
                use_wandb = False  # 发生错误后禁用wandb
        
        # 验证集评估
        if validation_loader is not None:
            val_loss, val_acc = evaluate(model, validation_loader, criterion, device, desc=f'Validating Epoch {epoch+1}')
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 如果使用wandb，记录验证指标
            if use_wandb:
                try:
                    wandb_log_dict.update({
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    })
                except Exception as e:
                    print(f"添加wandb验证指标时出错: {e}")
            
            # 打印当前学习率和验证结果
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
            
            # 早停逻辑
            if val_loss < best_val_loss:
                # 性能提升，保存模型状态
                print(f'验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}，保存模型状态')
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                early_stopping_counter = 0
                
                # 记录最佳验证指标
                if use_wandb:
                    try:
                        wandb_log_dict.update({
                            "best_val_loss": val_loss,
                            "best_val_accuracy": val_acc
                        })
                    except Exception as e:
                        print(f"记录wandb最佳验证指标时出错: {e}")
            else:
                early_stopping_counter += 1
                print(f'EarlyStopping 计数器: {early_stopping_counter}/{patience}')
                if early_stopping_counter >= patience:
                    print(f'Early stopping 在 epoch {epoch+1}')
                    # 恢复到最佳模型状态
                    model.load_state_dict(best_model_state)
                    if use_wandb:
                        try:
                            wandb_log_dict.update({
                                "early_stopping": epoch + 1
                            })
                        except Exception as e:
                            print(f"记录wandb早停信息时出错: {e}")
                    break
        else:
            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}')
        
        # 打印梯度范数和学习率信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"梯度范数: {epoch_grad_norm:.6f}, 学习率: {current_lr:.6f}")
        
        # 记录wandb指标
        if use_wandb:
            try:
                wandb.log(wandb_log_dict)
            except Exception as e:
                print(f"记录wandb指标时出错: {e}")
                use_wandb = False  # 发生错误后禁用wandb
        
        # 如果提供了学习率调度器，更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if validation_loader is not None else epoch_loss)  # 对于ReduceLROnPlateau，需要提供指标
            else:
                scheduler.step()  # 对于其他调度器，直接step
    
    # 如果没有验证集，则保存最后一个epoch的模型状态
    if validation_loader is None:
        best_model_state = model.state_dict().copy()
        
    if validation_loader is not None:
        return train_losses, train_accs, val_losses, val_accs, best_model_state
    else:
        return train_losses, train_accs, best_model_state

def evaluate(model, test_loader, criterion, device, desc='Evaluating'):
    """评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc=desc) as pbar:
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
    
    # 只有在描述为Evaluating时才打印
    if desc == 'Evaluating':
        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy 