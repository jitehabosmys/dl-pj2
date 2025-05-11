import argparse
import torch
import torch.nn as nn
import time
import sys
import os
import json
import itertools
from collections import OrderedDict

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_cifar_loader
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from utils.trainer import train, evaluate, set_seed
from utils.model_utils import count_parameters, save_model, get_optimizer, get_lr_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='超参数实验')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, 
                        choices=['BasicCNN', 'CNNWithBatchNorm', 'CNNWithDropout', 'ResNet'],
                        help='要训练的模型类型')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='ResNet的残差块数量')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                        help='Dropout的丢弃率')
    
    # 实验参数
    parser.add_argument('--exp_name', type=str, default='hyperparameter_experiment',
                        help='实验名称，用于区分不同实验')
    
    # 优化器实验
    parser.add_argument('--optimizers', type=str, nargs='+', 
                        choices=['adam', 'sgd', 'rmsprop', 'adamw', 'all'],
                        default=['adam'], help='要测试的优化器类型')
    
    # 学习率实验
    parser.add_argument('--learning_rates', type=float, nargs='+', 
                        default=[0.001], help='要测试的学习率列表')
    
    # 权重衰减实验
    parser.add_argument('--weight_decays', type=float, nargs='+', 
                        default=[0], help='要测试的权重衰减系数列表')
    
    # 学习率调度器实验
    parser.add_argument('--schedulers', type=str, nargs='+', 
                        choices=['step', 'cosine', 'reduce', 'none', 'all'],
                        default=['none'], help='要测试的学习率调度器类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=6, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='批量大小')
    
    # 自定义命名参数
    parser.add_argument('--result_prefix', type=str, default='',
                        help='结果文件名前缀，便于区分不同的实验系列')
    
    # 路径参数
    parser.add_argument('--output_dir', type=str, default='results/experiments',
                        help='输出目录')
    parser.add_argument('--download', action='store_true', default=True,
                        help='是否下载数据集')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='是否使用CUDA')
    parser.add_argument('--save_all_models', action='store_true', default=False,
                        help='是否保存所有模型，默认只保存最佳模型')
    
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

def generate_config_grid(args):
    """生成超参数组合网格"""
    # 处理优化器选择
    if 'all' in args.optimizers:
        optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
    else:
        optimizers = args.optimizers
    
    # 处理学习率调度器选择
    if 'all' in args.schedulers:
        schedulers = ['step', 'cosine', 'reduce', 'none']
    else:
        schedulers = args.schedulers
    
    # 生成所有超参数组合
    configs = []
    
    for opt, lr, wd, sched in itertools.product(
        optimizers, args.learning_rates, args.weight_decays, schedulers
    ):
        config = {
            'optimizer': opt,
            'learning_rate': lr,
            'weight_decay': wd,
            'scheduler': None if sched == 'none' else sched
        }
        configs.append(config)
    
    return configs

def run_experiment(config, model_name, args, device):
    """运行单个超参数配置的实验"""
    # 创建模型
    model = get_model(model_name)
    model = model.to(device)
    
    # 获取优化器
    optimizer = get_optimizer(
        model, 
        opt_name=config['optimizer'], 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # 获取学习率调度器
    scheduler = None
    if config['scheduler']:
        scheduler_kwargs = {
            'step_size': args.epochs // 3,
            'gamma': 0.1,
            'T_max': args.epochs,
            'patience': 1
        }
        scheduler = get_lr_scheduler(optimizer, config['scheduler'], **scheduler_kwargs)
    
    # 加载数据
    train_loader = get_cifar_loader(root='./data', train=True, 
                                   batch_size=args.batch_size, 
                                   download=args.download)
    test_loader = get_cifar_loader(root='./data', train=False, 
                                  batch_size=args.batch_size, 
                                  download=args.download)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    config_str = f"优化器:{config['optimizer']}, 学习率:{config['learning_rate']}, 权重衰减:{config['weight_decay']}, 调度器:{config['scheduler']}"
    print(f"\n{'='*80}\n实验配置: {config_str}\n{'='*80}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    train_losses, train_accs = train(
        model, train_loader, criterion, optimizer, device, 
        epochs=args.epochs, scheduler=scheduler
    )
    
    # 记录训练时间
    training_time = time.time() - start_time
    
    # 测试模型
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 构建结果字典
    result = {
        'model': model_name,
        'config': config,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'param_count': count_parameters(model),
        'training_time': training_time
    }
    
    # 生成配置描述符，用于文件名
    config_desc = f"{model_name}_{config['optimizer']}_lr{config['learning_rate']}_wd{config['weight_decay']}"
    if config['scheduler']:
        config_desc += f"_{config['scheduler']}"
    
    # 保存模型
    model_dir = os.path.join(args.output_dir, args.exp_name, 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_model(model, config_desc, save_dir=model_dir)
    
    return result, config_desc

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 生成超参数配置网格
    configs = generate_config_grid(args)
    print(f"实验总数: {len(configs)}")
    
    # 存储结果
    results = []
    
    # 运行每个配置的实验
    for i, config in enumerate(configs):
        print(f"\n实验 {i+1}/{len(configs)}")
        result, config_desc = run_experiment(config, args.model, args, device)
        results.append(result)
        
        # 打印实验结果
        print(f"\n配置: {config_desc}")
        print(f"测试准确率: {result['test_acc']:.2f}%")
        print(f"测试损失: {result['test_loss']:.4f}")
        print(f"训练时间: {result['training_time']:.2f}s")
    
    # 找出最佳配置
    best_result = max(results, key=lambda x: x['test_acc'])
    best_config = best_result['config']
    
    # 打印最佳结果
    print("\n" + "="*80)
    print(f"最佳配置:")
    print(f"模型: {args.model}")
    print(f"优化器: {best_config['optimizer']}")
    print(f"学习率: {best_config['learning_rate']}")
    print(f"权重衰减: {best_config['weight_decay']}")
    print(f"学习率调度器: {best_config['scheduler']}")
    print(f"测试准确率: {best_result['test_acc']:.2f}%")
    print(f"测试损失: {best_result['test_loss']:.4f}")
    print(f"训练时间: {best_result['training_time']:.2f}s")
    
    # 保存结果到JSON文件
    results_json = []
    for r in results:
        # 转换为可序列化的格式
        result_dict = {
            'model': r['model'],
            'config': r['config'],
            'test_acc': r['test_acc'],
            'test_loss': r['test_loss'],
            'train_losses': r['train_losses'],
            'train_accs': r['train_accs'],
            'param_count': r['param_count'],
            'training_time': r['training_time']
        }
        results_json.append(result_dict)
    
    # 添加最佳配置信息
    results_summary = {
        'experiment_name': args.exp_name,
        'model': args.model,
        'best_config': best_config,
        'best_test_acc': best_result['test_acc'],
        'results': results_json
    }
    
    # 保存到文件
    results_file = os.path.join(exp_dir, f"{args.model}_experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"实验结果已保存到 {results_file}")

if __name__ == "__main__":
    main() 