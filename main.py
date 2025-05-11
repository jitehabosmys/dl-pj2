import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os

# 导入自定义模块
from data.loaders import get_cifar_loader
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from utils.trainer import train, evaluate, set_seed
from utils.visualization import visualize_results, compare_models
from utils.model_utils import count_parameters, save_model, get_optimizer

def split_train_val_data(train_dataset, val_ratio=0.1, seed=42):
    """将训练集划分为训练集和验证集"""
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    val_size = int(val_ratio * dataset_size)
    
    # 设置随机种子确保可重复性
    random.seed(seed)
    random.shuffle(indices)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    return train_sampler, val_sampler

def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 确保结果目录存在
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)
    
    # 准备数据集和验证集
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载训练集
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 划分训练集和验证集
    val_ratio = 0.1  # 10%作为验证集
    train_sampler, val_sampler = split_train_val_data(train_dataset, val_ratio=val_ratio, seed=42)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, 
        sampler=train_sampler, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128,
        sampler=val_sampler, num_workers=2
    )
    
    print(f"训练集大小: {len(train_sampler)}, 验证集大小: {len(val_sampler)}")
    
    # 加载测试集
    test_loader = get_cifar_loader(root='./data', train=False, batch_size=128, download=True)
    
    # 模型配置
    models = {
        'BasicCNN': BasicCNN(),
        'CNNWithBatchNorm': CNNWithBatchNorm(),
        'CNNWithDropout': CNNWithDropout(dropout_rate=0.25),
        'ResNet': ResNet(num_blocks=2)
    }
    
    # 训练参数
    epochs = 6
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    patience = 5  # 早停耐心值
    
    # 存储结果
    results = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # 将模型移至设备
        model = model.to(device)
        
        # 选择优化器
        optimizer = get_optimizer(model, opt_name='adam', lr=learning_rate)
        
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',     # 监控验证损失最小化
            factor=0.1,     # 降低学习率的因子
            patience=2,     # 多少个epoch没有改善则降低学习率
            min_lr=1e-6     # 学习率下限
        )
        print(f"使用ReduceLROnPlateau学习率调度器，patience=2, factor=0.1")
        
        # 计算模型参数量
        param_count = count_parameters(model)
        print(f"Model has {param_count:.2f}M parameters")
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练模型
        train_results = train(
            model, train_loader, criterion, optimizer, device, 
            epochs=epochs, validation_loader=val_loader, patience=patience,
            scheduler=scheduler
        )
        
        # 提取训练结果
        train_losses, train_accs, val_losses, val_accs, best_model_state = train_results
        
        # 使用最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 可视化训练结果
        visualize_results(train_losses, train_accs, model_name, save_dir="results/images")
        
        # 保存模型
        save_model(model, model_name, save_dir="results/models")
        
        # 存储结果
        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'param_count': param_count,
            'training_time': training_time
        }
        
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best validation accuracy: {max(val_accs):.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
    
    # 比较模型性能
    compare_models(results, save_dir="results/images")
    
    # 输出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
    print(f"\nBest model: {best_model} with test accuracy {results[best_model]['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
