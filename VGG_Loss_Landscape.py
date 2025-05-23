import matplotlib as mpl
mpl.use('Agg')  # 非交互式后端，适合保存图像
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")  # 使用seaborn的网格风格
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import torchvision
import torchvision.transforms as transforms
import time
import argparse  # 添加argparse模块

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# 修改为更符合当前项目结构的路径
figures_path = 'results/images'
models_path = 'results/models'
loss_save_path = 'results'

# 确保目录存在
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(loss_save_path, exist_ok=True)

# Make sure you are using the right device.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

# 添加参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description='VGG Loss Landscape Analysis')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--learning_rates', type=str, 
                        default='1e-3,2e-3,1e-4,5e-4',
                        help='学习率列表，用逗号分隔 (默认: 1e-3,2e-3,1e-4,5e-4)')
    parser.add_argument('--seed', type=int, default=2020,
                        help='随机种子 (默认: 2020)')
    parser.add_argument('--skip_steps', type=int, default=25,
                        help='绘制损失景观时要跳过的初始步骤数 (默认: 25)')
    parser.add_argument('--plot_sample_rate', type=int, default=1,
                        help='可视化时每隔多少个点画一个（默认1，全部画）')
    args = parser.parse_args()
    
    # 将逗号分隔的字符串转换为浮点数列表
    args.learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    
    return args

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    """计算模型在给定数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []  # 所有步骤的损失列表
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 当前epoch的损失列表
        learning_curve[epoch] = 0  # 当前epoch的平均损失

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            # 记录损失值
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            
            loss.backward()
            optimizer.step()

        losses_list.append(loss_list)
        
        # 计算当前epoch的平均损失
        learning_curve[epoch] /= batches_n
        
        # 计算当前训练和验证准确率
        train_accuracy = get_accuracy(model, train_loader, device)
        train_accuracy_curve[epoch] = train_accuracy
        
        if val_loader is not None:
            val_accuracy = get_accuracy(model, val_loader, device)
            val_accuracy_curve[epoch] = val_accuracy
            
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                max_val_accuracy_epoch = epoch
                
            print(f"Epoch {epoch+1}/{epochs_n}, Loss: {learning_curve[epoch]:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Best Val Acc: {max_val_accuracy:.2f}% at epoch {max_val_accuracy_epoch+1}")
        else:
            print(f"Epoch {epoch+1}/{epochs_n}, Loss: {learning_curve[epoch]:.4f}, Train Acc: {train_accuracy:.2f}%")

    return losses_list

# 计算min_curve和max_curve
def compute_loss_curves(losses_lists):
    """计算多个模型的最小和最大损失曲线"""
    # losses_lists是一个三维列表: [模型][epoch][batch]
    # 需要将所有batch的损失展平为一个大列表
    
    if not losses_lists:
        return [], []
    
    try:
        # 将每个模型的所有batch损失值展平到一个列表中
        flattened_losses = []
        for i, model_losses in enumerate(losses_lists):
            # 展平每个模型的所有epoch的所有batch损失
            model_flat_losses = []
            for j, epoch_losses in enumerate(model_losses):
                if isinstance(epoch_losses, list):
                    model_flat_losses.extend(epoch_losses)
                else:
                    print(f"警告: 模型 {i+1}, epoch {j+1} 的数据不是列表")
            
            flattened_losses.append(model_flat_losses)
        
        # 找到最短的展平后损失列表长度
        if flattened_losses:
            lengths = [len(flat_losses) for flat_losses in flattened_losses]
            min_length = min(lengths)
        else:
            min_length = 0
            return [], []
        
        min_curve = []
        max_curve = []
        
        # 对每个batch步骤，找出所有模型中的最小和最大损失
        for step in range(min_length):
            step_losses = []
            for model_losses in flattened_losses:
                if step < len(model_losses):
                    step_losses.append(model_losses[step])
            
            if step_losses:
                min_curve.append(min(step_losses))
                max_curve.append(max(step_losses))
        
        return min_curve, max_curve
    except Exception as e:
        print(f"计算损失曲线时发生错误: {e}")
        # 返回空列表，避免程序崩溃
        return [], []

def main():
    """主函数：执行模型训练和损失景观分析"""
    # 解析命令行参数
    args = parse_args()
    epochs = args.epochs
    learning_rates = args.learning_rates
    skip_steps = args.skip_steps
    
    # 显示开始消息
    print(f"{'='*50}")
    print(f"开始训练和损失景观分析")
    print(f"{'='*50}")
    print(f"使用设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"学习率列表: {learning_rates}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(device.index)}")
    
    # 初始化数据加载器
    print("准备数据...")
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    # Display one batch of samples to ensure data loading works
    for X, y in train_loader:
        # Display sample batch information
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X min: {X.min():.4f}, X max: {X.max():.4f}")
        print(f"Label sample: {y[:10]}")
        break
    
    # 使用命令行参数中的epochs和学习率
    losses_lists_vgg = []
    losses_lists_vgg_bn = []

    # 对每个学习率训练VGG模型
    for lr in learning_rates:
        print(f"\n{'='*50}\n训练普通VGG模型 (学习率: {lr})\n{'='*50}")
        set_random_seeds(seed_value=args.seed, device=device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model_losses = train(model, optimizer, criterion, train_loader, test_loader, epochs_n=epochs)
        losses_lists_vgg.append(model_losses)
        
        # 保存该模型的所有batch损失
        # 将所有epoch的batch损失展平为一个列表
        flat_losses = []
        for epoch_losses in model_losses:
            flat_losses.extend(epoch_losses)
        np.savetxt(os.path.join(loss_save_path, f'loss_vgg_lr_{lr}.txt'), flat_losses, fmt='%.6f')

    # 对每个学习率训练VGG_BN模型
    for lr in learning_rates:
        print(f"\n{'='*50}\n训练带BN的VGG模型 (学习率: {lr})\n{'='*50}")
        set_random_seeds(seed_value=args.seed, device=device)
        model_bn = VGG_A_BatchNorm()
        optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
        criterion_bn = nn.CrossEntropyLoss()
        model_losses_bn = train(model_bn, optimizer_bn, criterion_bn, train_loader, test_loader, epochs_n=epochs)
        losses_lists_vgg_bn.append(model_losses_bn)
        
        # 保存该模型的所有batch损失
        # 将所有epoch的batch损失展平为一个列表
        flat_losses_bn = []
        for epoch_losses in model_losses_bn:
            flat_losses_bn.extend(epoch_losses)
        np.savetxt(os.path.join(loss_save_path, f'loss_vgg_bn_lr_{lr}.txt'), flat_losses_bn, fmt='%.6f')

    # 计算VGG模型的min_curve和max_curve
    min_curve_vgg, max_curve_vgg = compute_loss_curves(losses_lists_vgg)
    
    # 计算VGG_BN模型的min_curve和max_curve
    min_curve_vgg_bn, max_curve_vgg_bn = compute_loss_curves(losses_lists_vgg_bn)

    # 调试：打印前20个min_curve和max_curve的值
    print("VGG min_curve前20:", min_curve_vgg[:20])
    print("VGG max_curve前20:", max_curve_vgg[:20])
    print("VGG min==max前20:", [np.isclose(a, b) for a, b in zip(min_curve_vgg[:20], max_curve_vgg[:20])])

    print("VGG+BN min_curve前20:", min_curve_vgg_bn[:20])
    print("VGG+BN max_curve前20:", max_curve_vgg_bn[:20])
    print("VGG+BN min==max前20:", [np.isclose(a, b) for a, b in zip(min_curve_vgg_bn[:20], max_curve_vgg_bn[:20])])
    
    print(f"{'='*50}")
    print(f"训练完成，开始绘制损失景观对比图...")
    print(f"{'='*50}")
    
    print("准备绘制对比图...")
    
    try:
        # 确保数据为numpy一维数组
        min_curve_vgg = np.array(min_curve_vgg, dtype=float).flatten()
        max_curve_vgg = np.array(max_curve_vgg, dtype=float).flatten()
        min_curve_vgg_bn = np.array(min_curve_vgg_bn, dtype=float).flatten()
        max_curve_vgg_bn = np.array(max_curve_vgg_bn, dtype=float).flatten()
        
        # 截断前skip_steps个点
        min_curve_vgg = min_curve_vgg[skip_steps:]
        max_curve_vgg = max_curve_vgg[skip_steps:]
        min_curve_vgg_bn = min_curve_vgg_bn[skip_steps:]
        max_curve_vgg_bn = max_curve_vgg_bn[skip_steps:]
        
        # 采样
        sample_rate = args.plot_sample_rate
        steps_vgg = np.arange(len(min_curve_vgg))[::sample_rate]
        min_curve_vgg = min_curve_vgg[::sample_rate]
        max_curve_vgg = max_curve_vgg[::sample_rate]
        steps_vgg_bn = np.arange(len(min_curve_vgg_bn))[::sample_rate]
        min_curve_vgg_bn = min_curve_vgg_bn[::sample_rate]
        max_curve_vgg_bn = max_curve_vgg_bn[::sample_rate]
        
        print(f"已跳过前 {skip_steps} 个训练步骤，损失景观包含 {len(min_curve_vgg)} 个点")
        
        # 检查数组长度，确保至少有数据可以绘图
        if len(min_curve_vgg) > 0 and len(max_curve_vgg) > 0 and len(min_curve_vgg_bn) > 0 and len(max_curve_vgg_bn) > 0:
            plt.figure(figsize=(12, 8))
            
            # 只有学习率列表长度大于1时才填充，否则只画一条线
            if len(args.learning_rates) > 1:
                plt.fill_between(steps_vgg, min_curve_vgg, max_curve_vgg, 
                                alpha=0.35, color='#8FBC8F', label='Standard VGG')
                plt.fill_between(steps_vgg_bn, min_curve_vgg_bn, max_curve_vgg_bn, 
                                alpha=0.35, color='#DB7093', label='Standard VGG + BatchNorm')
            else:
                plt.plot(steps_vgg, min_curve_vgg, color='#8FBC8F', label='Standard VGG')
                plt.plot(steps_vgg_bn, min_curve_vgg_bn, color='#DB7093', label='Standard VGG + BatchNorm')
            
            plt.title('Loss Landscape', fontsize=18, pad=20)
            plt.xlabel('Steps', fontsize=14)
            plt.ylabel('Loss Value', fontsize=14)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True, alpha=0.3, color='gray')
            
            comparison_path = os.path.join(figures_path, "loss_landscape_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"已保存对比图到: {comparison_path}")
            plt.close()
        else:
            print("错误: 曲线数据为空，无法生成对比图")
    except Exception as e:
        print(f"生成损失景观对比图时出错: {e}")
        import traceback
        print(traceback.format_exc())

    print(f"{'='*50}")
    print("训练和可视化完成，损失景观图已保存。")
    print(f"可视化文件保存在: {figures_path}")
    print(f"{'='*50}")

# 只有当直接运行此脚本时才执行主函数
if __name__ == "__main__":
    main()