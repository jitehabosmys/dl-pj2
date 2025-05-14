import argparse
import torch
import torch.nn as nn
import time
import sys
import os
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BasicCNN, ResNet18, VGG_A, VGG_A_BatchNorm, PreActResNet18, get_pretrained_resnet18
from utils.trainer import train, evaluate, set_seed
from utils.model_utils import count_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='训练单个模型')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, 
                        choices=['BasicCNN', 'ResNet18', 'VGG_A', 'VGG_A_BatchNorm', 'PreActResNet18', 'PretrainedResNet18'],
                        help='要训练的模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例，设为0则不使用验证集')
    parser.add_argument('--patience', type=int, default=30, 
                        help='早停耐心值，连续多少个epoch验证性能未提升则停止训练')
    
    # 学习率参数
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='学习率')
    
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
    
    # wandb参数
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='是否使用wandb进行可视化')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb运行的名称，默认使用模型名称')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    
    # 预训练ResNet18特有参数
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='是否使用预训练权重(仅对PretrainedResNet18有效)')
    parser.add_argument('--finetune_mode', type=str, default='full',
                        choices=['full', 'last'],
                        help='微调模式: full(训练整个网络)或last(仅训练最后一层)')
    
    return parser.parse_args()

def get_model(model_name, pretrained=True, finetune_mode='full'):
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
    elif model_name == 'PretrainedResNet18':
        return get_pretrained_resnet18(pretrained=pretrained, finetune_mode=finetune_mode)
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
    
    # 初始化wandb（如果指定）
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            # 检测是否为Kaggle环境
            is_kaggle = os.path.exists("/kaggle/input")
            anonymous = None if not is_kaggle else "must"
            
            # 确定run名称
            run_name = args.wandb_run_name
            if run_name is None:
                run_name = args.model_name if args.model_name else args.model
                if args.exp_tag:
                    run_name = f"{run_name}_{args.exp_tag}"
            
            # 初始化wandb
            wandb.init(
                project="cifar-pj",
                name=run_name,
                config={
                    "model": args.model,
                    "optimizer": "sgd",  # 固定为SGD
                    "lr": args.lr,
                    "weight_decay": 5e-4,  # 固定为5e-4
                    "momentum": 0.9,  # 固定为0.9
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "validation_split": args.validation_split,
                    "patience": args.patience,
                    "exp_tag": args.exp_tag,
                    "pretrained": args.pretrained,
                    "finetune_mode": args.finetune_mode,
                    "lr_scheduler": "cosine"  # 固定为cosine
                },
                anonymous=anonymous
            )
            print(f"成功初始化wandb，项目名称: cifar-pj, 运行名称: {run_name}")
        except Exception as e:
            print(f"初始化wandb时出错: {e}")
            print("将继续训练但不使用wandb")
            use_wandb = False
    
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
    model = get_model(args.model, pretrained=args.pretrained, finetune_mode=args.finetune_mode)
    model = model.to(device)

    # 添加DataParallel支持
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        print("使用DataParallel进行多GPU训练")

    # 打印模型参数量
    param_count = count_parameters(model)
    print(f"模型有 {param_count:.2f}M 参数")
    
    # 记录模型架构到wandb
    if use_wandb:
        try:
            wandb.watch(model)
        except Exception as e:
            print(f"wandb.watch模型时出错: {e}")
    
    # 创建优化器（固定为SGD）
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                               momentum=0.9, weight_decay=5e-4)
    
    # 创建学习率调度器（固定为cosine）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"使用CosineAnnealingLR学习率调度器，T_max={args.epochs}")
    
    # 初始化最佳验证损失
    best_val_loss = float('inf')
    
    # 训练模型
    print(f"\n{'='*50}\n训练 {args.model}\n{'='*50}")
    start_time = time.time()
    
    # 定义保存模型的回调函数
    def save_best_model(model_state, epoch, val_loss, val_acc):
        # 生成自定义文件名
        if args.model_name:
            model_name = args.model_name
        else:
            model_name = args.model
        
        if args.exp_tag:
            model_name = f"{model_name}_{args.exp_tag}"
        
        # 保存最佳模型
        save_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, save_path)
        print(f"第{epoch+1}轮: 验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}，保存模型到 {save_path}")
        return save_path
    
    # 调用训练函数
    train_results = train(
        model, train_loader, criterion=nn.CrossEntropyLoss(), 
        optimizer=optimizer, device=device, epochs=args.epochs,
        validation_loader=val_loader,
        patience=args.patience,
        scheduler=scheduler,
        best_val_loss=best_val_loss,
        use_wandb=use_wandb,
        save_model_func=save_best_model if val_loader is not None else None
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
        # 如果没有验证集，则保存最后的模型
        model_name = args.model_name if args.model_name else args.model
        if args.exp_tag:
            model_name = f"{model_name}_{args.exp_tag}"
        save_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': best_model_state,
            'train_loss': train_losses[-1] if train_losses else 0,
            'train_acc': train_accs[-1] if train_accs else 0
        }, save_path)
        print(f"无验证集训练完成，保存模型到 {save_path}")
    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f} 秒")
    
    # 测试模型
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # 打印结果摘要
    print("\n结果摘要:")
    print(f"模型: {args.model}")
    if args.exp_tag:
        print(f"实验标签: {args.exp_tag}")
    print(f"优化器: SGD, 学习率: {args.lr}, 权重衰减: 5e-4, 动量: 0.9")
    print(f"学习率调度器: CosineAnnealing")
    print(f"参数量: {param_count:.2f}M")
    print(f"训练时间: {training_time:.2f}s")
    if val_loader is not None:
        print(f"最佳验证准确率: {max(val_accs):.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")
    print(f"模型已保存为: {model_name}.pth")
    
    # 记录最终测试结果到wandb
    if use_wandb:
        try:
            wandb.log({
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "training_time_seconds": training_time,
                "parameter_count_M": param_count
            })
            wandb.finish()
        except Exception as e:
            print(f"记录最终wandb结果时出错: {e}")


if __name__ == "__main__":
    main() 