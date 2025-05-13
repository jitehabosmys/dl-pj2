'''使用PyTorch训练CIFAR10'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import random
import numpy as np
import multiprocessing
import wandb  # 导入wandb

# 导入自定义模块
from data.loaders import get_cifar_loader
from models.preact_resnet18 import PreActResNet18
# 导入更多模型
from models import BasicCNN, ResNet18, VGG_A, VGG_A_BatchNorm, PretrainedResNet18, get_pretrained_resnet18
from utils.model_utils import count_parameters, save_model

# 定义进度条函数
def progress_bar(current, total, msg=None):
    """简单的进度条实现"""
    if msg:
        print(f'[{current}/{total}] {msg}', end='\r')
    else:
        print(f'[{current}/{total}]', end='\r')

# 参数解析
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--validation_split', type=float, default=0.1, help='validation set ratio (0 to use all data for training)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=200, help='total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience (number of epochs)')
parser.add_argument('--save_name', type=str, default='ckpt', help='保存的模型文件名，不需要添加.pth后缀')
# 添加模型选择参数
parser.add_argument('--model', type=str, default='PreActResNet18', 
                    choices=['BasicCNN', 'ResNet18', 'VGG_A', 'VGG_A_BatchNorm', 'PreActResNet18', 'PretrainedResNet18'],
                    help='要训练的模型类型')
# 添加预训练模型参数
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='是否使用预训练权重(仅对PretrainedResNet18有效)')
parser.add_argument('--finetune_mode', type=str, default='full',
                    choices=['full', 'last'],
                    help='微调模式: full(训练整个网络)或last(仅训练最后一层)')
# 添加wandb参数
parser.add_argument('--use_wandb', action='store_true', default=False,
                    help='是否使用wandb进行可视化')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='wandb运行名称，默认使用模型名称')
args = parser.parse_args()

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # 最佳测试准确率
start_epoch = 0  # 从第0个epoch开始或者从检查点恢复

# 设置随机种子函数
def set_seed(seed=42):
    """设置所有可能的随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

# 设置随机种子
set_seed(args.seed)

# 数据准备
print('==> 准备数据..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# 处理验证集划分
train_loader = None
val_loader = None

if args.validation_split > 0:
    # 划分训练集和验证集
    from torch.utils.data.sampler import SubsetRandomSampler
    
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    val_size = int(args.validation_split * dataset_size)
    
    # 设置随机种子确保可重复性
    random.seed(args.seed)
    random.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, 
        sampler=train_sampler, num_workers=2)
        
    val_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=2)
    
    print(f'训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}')
else:
    # 不使用验证集，全部数据用于训练
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f'使用全部训练数据，无验证集')

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 构建模型
print('==> 构建模型..')

# 根据参数选择模型
def get_model(model_name, pretrained=False, finetune_mode='full'):
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

# 创建模型实例
net = get_model(args.model, pretrained=args.pretrained, finetune_mode=args.finetune_mode)
print(f'使用模型: {args.model}')

net = net.to(device)
print(f"使用模型: {args.model}")

# 计算模型参数量
param_count = count_parameters(net)
print(f"模型有 {param_count:.2f}M 参数")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # 从检查点恢复
    print('==> 从检查点恢复..')
    assert os.path.isdir('checkpoint'), 'Error: 未找到checkpoint目录!'
    checkpoint = torch.load(f'./checkpoint/{args.save_name}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
print(f"使用CosineAnnealingLR学习率调度器，T_max={args.epochs}")

# 初始化wandb（如果指定）
if args.use_wandb:
    try:
        # 检测是否为Kaggle环境
        is_kaggle = os.path.exists("/kaggle/input")
        anonymous = None
        
        # 设置wandb环境变量（适用于Kaggle）
        if is_kaggle:
            os.environ["WANDB_CONSOLE"] = "off"  # 在Kaggle上禁用特殊的console输出
            os.environ["WANDB_SILENT"] = "true"  # 减少一些非必要输出
            print("检测到Kaggle环境，尝试从secrets获取wandb API密钥")
            
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                secret_value = user_secrets.get_secret("wandb_api")
                wandb.login(key=secret_value)
                print("成功从Kaggle secrets获取wandb API密钥")
            except Exception as e:
                print(f"无法从Kaggle secrets获取wandb API密钥: {e}")
                print("如果要使用您的W&B账户，请前往Kaggle的Add-ons -> Secrets，提供您的W&B访问令牌。使用标签名称'wandb_api'。")
                print("从这里获取您的W&B访问令牌: https://wandb.ai/authorize")
                anonymous = "must"
        
        # 确定run名称
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = args.model
            run_name = f"{run_name}_{args.save_name}" if args.save_name != 'ckpt' else run_name
        
        # 初始化wandb
        wandb.init(
            project='cifar-pj',  # 固定项目名为cifar-pj
            name=run_name,
            config={
                "model": args.model,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "validation_split": args.validation_split,
                "patience": args.patience,
                "resume": args.resume,
                "save_name": args.save_name,
                "pretrained": args.pretrained,
                "finetune_mode": args.finetune_mode if args.model == 'PretrainedResNet18' else None,
                "is_kaggle": is_kaggle if 'is_kaggle' in locals() else False
            },
            anonymous=anonymous
        )
        print(f"成功初始化wandb，项目名称: cifar-pj, 运行名称: {run_name}")
        
        # 记录模型架构到wandb
        wandb.watch(net)
    except Exception as e:
        print(f"初始化wandb时出错: {e}")
        print("将继续训练但不使用wandb")
        args.use_wandb = False

# 训练
def train(epoch, net, train_loader, criterion, optimizer, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    train_acc = 100.*correct/total
    epoch_time = time.time() - epoch_start_time
    print(f'训练耗时: {epoch_time:.2f}s | 训练损失: {train_loss/(batch_idx+1):.4f} | 训练准确率: {train_acc:.2f}% | 学习率: {current_lr:.6f}')
    
    # 记录wandb指标
    if args.use_wandb:
        try:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss/(batch_idx+1),
                "train_accuracy": train_acc,
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            })
        except Exception as e:
            print(f"记录wandb指标时出错: {e}")
    
    return train_loss/(batch_idx+1), train_acc

# 验证
def validate(epoch, net, val_loader, criterion, device):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    val_acc = 100.*correct/total
    print(f'验证损失: {val_loss/(batch_idx+1):.4f} | 验证准确率: {val_acc:.2f}%')
    
    # 记录wandb指标
    if args.use_wandb:
        try:
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss/(batch_idx+1),
                "val_accuracy": val_acc
            })
        except Exception as e:
            print(f"记录wandb验证指标时出错: {e}")
    
    return val_loss/(batch_idx+1), val_acc

# 测试
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100.*correct/total
    print(f'测试损失: {test_loss/(batch_idx+1):.4f} | 测试准确率: {test_acc:.2f}%')
    
    # 记录最终测试指标到wandb
    if args.use_wandb:
        try:
            wandb.log({
                "test_loss": test_loss/(batch_idx+1),
                "test_accuracy": test_acc
            })
        except Exception as e:
            print(f"记录wandb测试指标时出错: {e}")
    
    return test_loss/(batch_idx+1), test_acc

def main():
    print(f"\n{'='*50}\n训练 {args.model}\n{'='*50}")
    start_time = time.time()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 用于早停的变量
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0

    for epoch in range(start_epoch, start_epoch+args.epochs):
        # 训练
        train_loss, train_acc = train(epoch, net, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        if val_loader is not None:
            val_loss, val_acc = validate(epoch, net, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 保存最佳模型（基于验证损失）
            if val_loss < best_val_loss:
                print(f'验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}，保存模型状态')
                best_val_loss = val_loss
                best_model_state = net.state_dict().copy()
                
                # 保存检查点
                state = {
                    'net': net.state_dict(),
                    'acc': val_acc,
                    'loss': val_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                save_path = f'./checkpoint/{args.save_name}.pth'
                torch.save(state, save_path)
                
                # 重置早停计数器
                early_stopping_counter = 0
            else:
                # 验证损失没有改善，增加早停计数
                early_stopping_counter += 1
                print(f'EarlyStopping 计数器: {early_stopping_counter}/{args.patience}')
                
                # 如果连续args.patience个epoch验证损失没有改善，停止训练
                if early_stopping_counter >= args.patience:
                    print(f'Early stopping 在 epoch {epoch+1}')
                    # 恢复到最佳模型状态
                    net.load_state_dict(best_model_state)
                    break
        else:
            # 如果没有验证集，保存每个epoch的模型
            state = {
                'net': net.state_dict(),
                'acc': train_acc,
                'loss': train_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_path = f'./checkpoint/{args.save_name}.pth'
            torch.save(state, save_path)
            best_model_state = net.state_dict().copy()
        
        scheduler.step()

    total_time = time.time() - start_time
    print(f'\n训练完成! 总耗时: {total_time:.2f}秒')

    # 确保使用最佳模型进行测试
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    # 在训练结束后用最佳模型测试一次
    print('\n使用最佳模型在测试集上评估')
    test_loss, test_acc = test(net, test_loader, criterion, device)

    if val_loader is not None:
        max_val_acc = max(val_accs)
        print(f'最佳验证准确率: {max_val_acc:.2f}%')

    print(f'最佳测试准确率: {test_acc:.2f}%')
    print(f"模型保存在: ./checkpoint/{args.save_name}.pth")

    # 关闭wandb
    if args.use_wandb:
        try:
            # 记录最终的训练时间和参数量
            param_count = count_parameters(net)
            wandb.log({
                "total_training_time": total_time,
                "parameter_count_M": param_count
            })
            wandb.finish()
        except Exception as e:
            print(f"关闭wandb时出错: {e}")

if __name__ == "__main__":
    # 在Windows上需要添加这一行以支持多进程
    multiprocessing.freeze_support()
    main()
