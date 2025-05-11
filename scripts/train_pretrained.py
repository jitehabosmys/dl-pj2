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
from models.pretrained_models import get_pretrained_resnet18
from utils.trainer import train, evaluate, set_seed
from utils.visualization import visualize_results
from utils.model_utils import count_parameters, save_model, get_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='训练预训练ResNet18模型')
    
    # 模型参数
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='是否加载预训练权重')
    parser.add_argument('--finetune_mode', type=str, default='full',
                        choices=['full', 'last_layer', 'partial'],
                        help='微调模式: full-全部参数, last_layer-仅最后层, partial-部分层')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例，设为0则不使用验证集')
    parser.add_argument('--patience', type=int, default=5, 
                        help='早停耐心值，连续多少个epoch验证性能未提升则停止训练')
    
    # 优化参数
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数（L2正则化）')
    parser.add_argument('--lr_scheduler', action='store_true', default=True,
                        help='是否使用学习率调度器(ReduceLROnPlateau)')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='学习率调度器的耐心值，验证损失多少个epoch未改善则降低学习率')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='学习率调度器的降低因子')
    
    # 自定义命名参数
    parser.add_argument('--model_name', type=str, default=None,
                        help='保存模型的自定义名称，默认基于微调模式自动生成')
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
    
    # 标准化参数
    # ResNet预训练模型通常使用ImageNet均值和标准差
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 测试/验证集变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # 加载训练集
    train_dataset = CIFAR10(root='./data', train=True, download=args.download, transform=train_transform)
    
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
    test_dataset = CIFAR10(root='./data', train=False, download=args.download, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2
    )
    
    # 创建预训练ResNet18模型
    model = get_pretrained_resnet18(
        num_classes=10, 
        pretrained=args.pretrained,
        finetune_mode=args.finetune_mode
    )
    model = model.to(device)
    
    # 打印模型参数量和可训练参数量
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数量: {total_params:.2f}M")
    print(f"可训练参数量: {trainable_params:.2f}M")
    
    # 创建优化器 - 只优化可训练参数
    if args.finetune_mode != 'full':
        trainable_params = model.get_trainable_parameters()
        optimizer = get_optimizer(trainable_params, args.optimizer, args.lr, args.weight_decay)
    else:
        optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # 创建学习率调度器
    scheduler = None
    if args.lr_scheduler and args.validation_split > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',               # 监控验证损失最小化
            factor=args.lr_factor,    # 降低学习率的因子
            patience=args.lr_patience,# 多少个epoch没有改善则降低学习率
            min_lr=1e-6               # 学习率下限
        )
        print(f"使用ReduceLROnPlateau学习率调度器，patience={args.lr_patience}, factor={args.lr_factor}")
    
    # 训练模型
    print(f"\n{'='*50}\n训练预训练ResNet18 (微调模式: {args.finetune_mode})\n{'='*50}")
    start_time = time.time()
    
    # 调用训练函数
    train_results = train(
        model, train_loader, criterion=nn.CrossEntropyLoss(), 
        optimizer=optimizer, device=device, epochs=args.epochs,
        validation_loader=val_loader,
        patience=args.patience,
        scheduler=scheduler
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
        model_name = f"resnet18_pretrained_{args.finetune_mode}"
    
    if args.exp_tag:
        model_name = f"{model_name}_{args.exp_tag}"
        result_prefix = f"resnet18_{args.finetune_mode}_{args.exp_tag}"
    else:
        result_prefix = f"resnet18_{args.finetune_mode}"
    
    # 可视化训练结果
    visualize_results(train_losses, train_accs, result_prefix, save_dir=image_dir)
    
    # 保存模型
    save_model(model, model_name, save_dir=model_dir)
    
    # 打印结果摘要
    print("\n结果摘要:")
    print(f"模型: ResNet18预训练 ({args.finetune_mode}模式微调)")
    if args.exp_tag:
        print(f"实验标签: {args.exp_tag}")
    print(f"优化器: {args.optimizer}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"总参数量: {total_params:.2f}M")
    print(f"可训练参数量: {trainable_params:.2f}M")
    print(f"训练时间: {training_time:.2f}s")
    if val_loader is not None:
        print(f"最佳验证准确率: {max(val_accs):.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")
    print(f"模型保存为: {model_name}.pth")

if __name__ == "__main__":
    main() 