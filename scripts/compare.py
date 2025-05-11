import argparse
import torch
import torch.nn as nn
import sys
import os
import json
from collections import OrderedDict

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_cifar_loader
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from utils.trainer import evaluate, set_seed
from utils.visualization import compare_models
from utils.model_utils import load_model, count_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='比较多个预训练模型的性能')
    
    # 比较参数
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['BasicCNN', 'CNNWithBatchNorm', 'CNNWithDropout', 'ResNet', 'all'],
                        default=['all'], help='要比较的模型类型，使用all表示所有模型')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    
    # 自定义命名参数
    parser.add_argument('--comparison_name', type=str, default='model_comparison',
                        help='比较结果的自定义名称，用于区分不同的比较实验')
    parser.add_argument('--model_names', type=str, nargs='+', default=None,
                        help='要加载的各模型的自定义文件名，顺序应与models参数对应')
    
    # 路径参数
    parser.add_argument('--model_dir', type=str, default='results/models',
                        help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--download', action='store_true', default=True,
                        help='是否下载数据集')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    parser.add_argument('--save_results', action='store_true', default=False,
                        help='是否保存比较结果到JSON文件')
    
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
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    image_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # 要比较的模型列表
    if 'all' in args.models:
        model_names = ['BasicCNN', 'CNNWithBatchNorm', 'CNNWithDropout', 'ResNet']
    else:
        model_names = args.models
    
    # 加载测试数据
    test_loader = get_cifar_loader(root='./data', train=False, 
                                  batch_size=args.batch_size, 
                                  download=args.download)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 存储结果
    results = OrderedDict()
    
    # 测试每个模型
    for i, model_name in enumerate(model_names):
        print(f"\n{'='*50}\n测试 {model_name}\n{'='*50}")
        
        # 创建模型实例
        model = get_model(model_name)
        
        # 获取模型文件名：如果提供了自定义名称，则使用；否则使用模型类型名称
        model_file_name = model_name
        if args.model_names is not None and i < len(args.model_names):
            model_file_name = args.model_names[i]
            print(f"使用自定义模型名称: {model_file_name}")
        
        # 加载预训练权重
        load_success = load_model(model, model_file_name, save_dir=args.model_dir)
        if not load_success:
            print(f"无法加载模型 {model_file_name}，跳过该模型")
            continue
        
        # 将模型移至设备并设置为评估模式
        model = model.to(device)
        model.eval()
        
        # 计算模型参数量
        param_count = count_parameters(model)
        print(f"模型有 {param_count:.2f}M 参数")
        
        # 在测试集上评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 存储结果 - 使用带索引的键以避免同名模型覆盖
        result_key = f"{model_name}_{i}"
        if args.model_names is not None and i < len(args.model_names):
            result_key = f"{model_name}_{args.model_names[i]}"
        
        results[result_key] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'param_count': param_count,
            'display_name': model_file_name,  # 添加显示名称
            # 由于是加载预训练模型，不需要训练时间，为了兼容visualize.compare_models()函数，添加一个0
            'training_time': 0
        }
    
    # 如果没有成功加载任何模型，退出
    if not results:
        print("没有成功加载任何模型，请先训练模型")
        return
    
    # 打印比较结果
    print("\n比较结果:")
    print(f"{'模型名称':<20} {'参数量(M)':<12} {'测试准确率(%)':<15} {'测试损失':<10}")
    print("-" * 60)
    
    # 为了正确显示，创建新的显示结果字典
    display_results = {}
    for key, result in results.items():
        display_name = result.get('display_name', key)
        display_results[display_name] = result
    
    for name, result in display_results.items():
        print(f"{name:<20} {result['param_count']:<12.2f} {result['test_acc']:<15.2f} {result['test_loss']:<10.4f}")
    
    # 找出最佳模型
    best_key = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
    best_display_name = results[best_key].get('display_name', best_key)
    best_acc = results[best_key]['test_acc']
    print(f"\n最佳模型: {best_display_name} (测试准确率: {best_acc:.2f}%)")
    
    # 修改结果字典以便可视化时使用display_name作为键
    viz_results = {}
    for key, result in results.items():
        display_name = result.get('display_name', key)
        viz_results[display_name] = result
    
    # 可视化比较结果
    compare_models(viz_results, save_dir=image_dir, comparison_name=args.comparison_name)
    
    # 保存结果到JSON文件
    if args.save_results:
        # 将OrderedDict转换为普通dict以便JSON序列化
        results_json = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}
        
        # 添加最佳模型信息
        results_json['best_model'] = best_key
        results_json['best_model_display'] = best_display_name
        
        # 保存到文件
        results_file = os.path.join(args.output_dir, f'{args.comparison_name}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f"比较结果已保存到 {results_file}")

if __name__ == "__main__":
    main() 