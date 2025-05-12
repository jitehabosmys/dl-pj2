import argparse
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_cifar_loader
from models import BasicCNN, ResNet18, VGG_A, VGG_A_BatchNorm, PreActResNet18
from utils.trainer import evaluate, set_seed
from utils.model_utils import load_model, count_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='测试预训练模型')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, 
                        choices=['BasicCNN', 'ResNet18', 'VGG_A', 'VGG_A_BatchNorm', 'PreActResNet18'],
                        help='要测试的模型类型')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    
    # 自定义命名参数
    parser.add_argument('--model_name', type=str, default=None,
                        help='要加载的模型的文件名（不包含.pth），默认使用模型类型')
    parser.add_argument('--result_tag', type=str, default='',
                        help='测试结果的标签，会添加到输出中')
    
    # 路径参数
    parser.add_argument('--model_dir', type=str, default='results/models',
                        help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录（如果需要保存测试结果）')
    parser.add_argument('--download', action='store_true', default=True,
                        help='是否下载数据集')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='是否输出详细信息')
    parser.add_argument('--save_results', action='store_true', default=False,
                        help='是否保存测试结果到文件')
    
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

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据集和数据加载器
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    
    # CIFAR-10精确的归一化参数
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载测试集
    test_dataset = CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2
    )
    
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = args.model
    
    # 创建模型实例
    model = get_model(args.model)
    
    # 加载预训练权重
    load_success = load_model(model, model_name, save_dir=args.model_dir)
    if not load_success:
        print(f"无法加载模型，请先训练模型或检查模型文件路径。")
        return
    
    # 将模型移至设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 如果需要详细信息，则打印模型参数量
    if args.verbose:
        param_count = count_parameters(model)
        print(f"模型有 {param_count:.2f}M 参数")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 在测试集上评估模型
    print(f"\n{'='*50}\n测试 {args.model}\n{'='*50}")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 打印结果
    print("\n测试结果:")
    print(f"模型: {args.model}")
    if args.result_tag:
        print(f"结果标签: {args.result_tag}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")

if __name__ == "__main__":
    main() 