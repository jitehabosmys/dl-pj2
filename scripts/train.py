import argparse
import torch
import torch.nn as nn
import time
import sys
import os
import random

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_cifar_loader
from models import BasicCNN, ResNet18, VGG_A, VGG_A_BatchNorm, PreActResNet18
from utils.trainer import train, evaluate, set_seed
from utils.visualization import visualize_results
from utils.model_utils import count_parameters, save_model, get_optimizer, load_model

def parse_args():
    parser = argparse.ArgumentParser(description='训练单个模型')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, 
                        choices=['BasicCNN', 'ResNet18', 'VGG_A', 'VGG_A_BatchNorm', 'PreActResNet18'],
                        help='要训练的模型类型')
    
    # 加载预训练模型参数
    parser.add_argument('--resume', type=str, default=None,
                        help='指定要加载的预训练模型文件名（不包含.pth扩展名），用于继续训练')
    parser.add_argument('--model_dir', type=str, default='results/models',
                        help='模型加载和保存目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, 
                        help='训练轮数（如果使用--resume，则表示额外训练的轮数）')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例，设为0则不使用验证集')
    parser.add_argument('--patience', type=int, default=20, 
                        help='早停耐心值，连续多少个epoch验证性能未提升则停止训练')
    
    # 优化参数
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减系数（L2正则化）')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量值(仅用于SGD优化器)')
    
    # 自定义命名参数
    parser.add_argument('--model_name', type=str, default=None,
                        help='保存模型的自定义名称，默认使用模型类型')
    parser.add_argument('--exp_tag', type=str, default='',
                        help='实验标签，会添加到保存文件名中，便于区分不同实验')
    
    # 路径参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--download', action='store_true', default=True,
                        help='是否下载数据集')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    
    return parser.parse_args()

def get_model(model_name):
    """根据名称创建模型实例"""
    if model_name == 'BasicCNN':
        return BasicCNN()
    elif model_name == 'ResNet18':
        return ResNet18()
    elif model_name == 'VGG_A':
        return VGG_A()
    elif model_name == 'VGG_A_BatchNorm':
        return VGG_A_BatchNorm()
    elif model_name == 'PreActResNet18':
        return PreActResNet18()
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

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
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    model_dir = os.path.join(args.output_dir, 'models')
    image_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据集和数据加载器
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    
    # CIFAR-10精确的归一化参数
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 添加数据增强
        transforms.RandomHorizontalFlip(),      # 添加数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # 加载训练集
    train_dataset = CIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
    
    # 是否划分验证集
    train_loader = None
    val_loader = None
    
    if args.validation_split > 0:
        # 划分训练集和验证集
        train_sampler, val_sampler = split_train_val_data(train_dataset, args.validation_split, args.seed)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            sampler=train_sampler, num_workers=2
        )
        
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=val_sampler, num_workers=2
        )
        
        print(f"训练集大小: {len(train_sampler)}, 验证集大小: {len(val_sampler)}")
    else:
        # 不使用验证集
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=2
        )
    
    # 加载测试集
    test_dataset = CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2
    )
    
    # 创建模型
    model = get_model(args.model)
    model = model.to(device)
    
    # 打印模型参数量
    param_count = count_parameters(model)
    print(f"模型有 {param_count:.2f}M 参数")
    
    # 创建优化器（根据参数）
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                   momentum=args.momentum, 
                                   weight_decay=args.weight_decay)
    else:
        optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # 初始epoch和最佳准确率
    start_epoch = 0
    best_acc = 0
    best_val_loss = float('inf')
    
    # 创建学习率调度器 - 使用CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"使用CosineAnnealingLR学习率调度器，T_max={args.epochs}")
    
    # 如果指定了预训练模型，则加载
    if args.resume:
        print(f"尝试加载预训练模型: {args.resume}")
        load_success, checkpoint = load_model(model, args.resume, optimizer, scheduler, save_dir=args.model_dir)
        if load_success:
            if checkpoint is not None:
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                    print(f"继续从epoch {start_epoch}开始训练")
                if 'acc' in checkpoint:
                    best_acc = checkpoint['acc']
                    print(f"加载的最佳准确率: {best_acc:.2f}%")
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                    print(f"加载的最佳验证损失: {best_val_loss:.4f}")
                
                # 在resume模式下，更新学习率调度器的T_max
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.T_max = start_epoch + args.epochs
                    print(f"更新学习率调度器：T_max={scheduler.T_max}")
            print(f"成功加载预训练模型，继续训练...")
        else:
            print(f"无法加载预训练模型，将从头开始训练...")
    else:
        # 非resume模式，设置学习率调度器的T_max
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.T_max = args.epochs
    
    # 训练模型
    print(f"\n{'='*50}\n训练 {args.model}\n{'='*50}")
    start_time = time.time()
    
    # 调用训练函数
    train_results = train(
        model, train_loader, criterion=nn.CrossEntropyLoss(), 
        optimizer=optimizer, device=device, epochs=args.epochs,  # 不再减去start_epoch，直接使用args.epochs
        validation_loader=val_loader,
        patience=args.patience,
        scheduler=scheduler,
        best_val_loss=best_val_loss  # 传入已有的最佳验证损失
    )
    
    # 提取训练结果
    if val_loader is not None:
        train_losses, train_accs, val_losses, val_accs, best_model_state = train_results
        # 使用最佳模型状态
        model.load_state_dict(best_model_state)
    else:
        train_losses, train_accs, best_model_state = train_results
        # 使用最佳模型状态
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f} 秒")
    
    # 测试模型
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # 生成自定义文件名
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = args.model
    
    if args.exp_tag:
        model_name = f"{model_name}_{args.exp_tag}"
        result_prefix = f"{args.model}_{args.exp_tag}"
    else:
        result_prefix = args.model_name if args.model_name else args.model
    
    # 可视化训练结果
    visualize_results(train_losses, train_accs, result_prefix, save_dir=image_dir)
    
    # 计算当前训练完的总epoch数
    current_epoch = start_epoch + len(train_losses) - 1
    
    # 获取最终的最佳验证损失
    final_best_val_loss = best_val_loss
    if val_loader is not None and len(val_losses) > 0:
        min_val_loss_idx = val_losses.index(min(val_losses))
        final_best_val_loss = val_losses[min_val_loss_idx]
    
    # 保存模型
    save_model(model, model_name, optimizer, scheduler, current_epoch, test_acc, final_best_val_loss, save_dir=model_dir)
    
    # 打印结果摘要
    print("\n结果摘要:")
    print(f"模型: {args.model}")
    if args.resume:
        print(f"从预训练模型继续: {args.resume}")
    if args.exp_tag:
        print(f"实验标签: {args.exp_tag}")
    print(f"优化器: {args.optimizer}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"参数量: {param_count:.2f}M")
    print(f"训练时间: {training_time:.2f}s")
    if val_loader is not None:
        print(f"最佳验证准确率: {max(val_accs):.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")
    print(f"模型保存为: {model_name}.pth")

if __name__ == "__main__":
    main() 