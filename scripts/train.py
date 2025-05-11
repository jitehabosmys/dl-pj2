import argparse
import torch
import torch.nn as nn
import time
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_cifar_loader
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from utils.trainer import train, evaluate, set_seed
from utils.visualization import visualize_results
from utils.model_utils import count_parameters, save_model, get_optimizer, get_lr_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='训练单个模型')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, 
                        choices=['BasicCNN', 'CNNWithBatchNorm', 'CNNWithDropout', 'ResNet'],
                        help='要训练的模型类型')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='ResNet的残差块数量')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                        help='Dropout的丢弃率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=6, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    
    # 优化参数
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'],
                        help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='权重衰减系数（L2正则化）')
    
    # 自定义命名参数
    parser.add_argument('--model_name', type=str, default=None,
                        help='保存模型的自定义名称，默认使用模型类型')
    parser.add_argument('--exp_tag', type=str, default='',
                        help='实验标签，会添加到保存文件名中，便于区分不同实验')
    
    # 路径参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--download', action='store_true', default=False,
                        help='是否下载数据集')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    
    return parser.parse_args()

def get_model(model_name, num_blocks=2, dropout_rate=0.25):
    """根据名称创建模型实例"""
    if model_name == 'BasicCNN':
        return BasicCNN()
    elif model_name == 'CNNWithBatchNorm':
        return CNNWithBatchNorm()
    elif model_name == 'CNNWithDropout':
        return CNNWithDropout(dropout_rate=dropout_rate)
    elif model_name == 'ResNet':
        return ResNet(num_blocks=num_blocks)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

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
    
    # 加载数据
    train_loader = get_cifar_loader(root='./data', train=True, 
                                   batch_size=args.batch_size, 
                                   download=args.download)
    test_loader = get_cifar_loader(root='./data', train=False, 
                                  batch_size=args.batch_size, 
                                  download=args.download)
    
    # 创建模型
    model = get_model(args.model, args.num_blocks, args.dropout_rate)
    model = model.to(device)
    
    # 打印模型参数量
    param_count = count_parameters(model)
    print(f"模型有 {param_count:.2f}M 参数")
    
    # 创建优化器
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # 训练模型
    print(f"\n{'='*50}\n训练 {args.model}\n{'='*50}")
    start_time = time.time()
    
    train_losses, train_accs = train(
        model, train_loader, criterion=nn.CrossEntropyLoss(), 
        optimizer=optimizer, device=device, epochs=args.epochs
    )
    
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
        result_prefix = args.model_name
    
    # 可视化训练结果
    visualize_results(train_losses, train_accs, result_prefix, save_dir=image_dir)
    
    # 保存模型
    save_model(model, model_name, save_dir=model_dir)
    
    # 打印结果摘要
    print("\n结果摘要:")
    print(f"模型: {args.model}")
    if args.exp_tag:
        print(f"实验标签: {args.exp_tag}")
    print(f"优化器: {args.optimizer}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"参数量: {param_count:.2f}M")
    print(f"训练时间: {training_time:.2f}s")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")
    print(f"模型保存为: {model_name}.pth")

if __name__ == "__main__":
    main() 