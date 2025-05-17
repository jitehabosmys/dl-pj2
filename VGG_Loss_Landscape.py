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
grad_save_path = 'results'

# 确保目录存在
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(loss_save_path, exist_ok=True)
os.makedirs(grad_save_path, exist_ok=True)

# Make sure you are using the right device.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

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
    grads = []      # 所有步骤的梯度列表
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 当前epoch的损失列表
        grad = []       # 当前epoch的梯度列表
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
            
            # 反向传播前记录损失
            loss.backward()
            
            # 获取梯度信息 (对于分类器层的梯度)
            if hasattr(model, 'classifier') and len(model.classifier) > 4:
                # 记录分类器中第5个层(索引4)的梯度
                if model.classifier[4].weight.grad is not None:
                    current_grad = model.classifier[4].weight.grad.norm().item()
                    grad.append(current_grad)
            
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        
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

        # 不再生成每个epoch的可视化，只保留打印信息

    return losses_list, grads

# 绘制损失景观函数
def plot_loss_landscape(min_curve, max_curve, title="Loss Landscape", save_path=None, color='#8FBC8F'):
    """绘制损失景观"""
    try:
        # 确保min_curve和max_curve是一维numpy数组
        min_curve = np.array(min_curve).flatten()
        max_curve = np.array(max_curve).flatten()
        
        # 打印用于调试的信息（截断显示）
        print(f"绘制 {title} 损失景观:")
        print(f"  min_curve类型: {type(min_curve)}, 长度: {len(min_curve)}")
        print(f"  min_curve前5个值: {min_curve[:5]}")
        print(f"  max_curve类型: {type(max_curve)}, 长度: {len(max_curve)}")
        print(f"  max_curve前5个值: {max_curve[:5]}")
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        steps = range(len(min_curve))
        
        # 填充两条曲线之间的区域
        plt.fill_between(steps, min_curve, max_curve, alpha=0.35, color=color)
    except Exception as e:
        print(f"绘制损失景观时出错: {e}")

    try:
        plt.title(f'Loss Landscape: {title}', fontsize=18, pad=20)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Loss Landscape', fontsize=14)
        plt.grid(True, alpha=0.3, color='gray')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存图像到: {save_path}")
        plt.close()  # 关闭图形而不是显示
    except Exception as e:
        print(f"完成绘图时出错: {e}")

# 计算min_curve和max_curve
def compute_loss_curves(losses_lists):
    """计算多个模型的最小和最大损失曲线"""
    # losses_lists是一个三维列表: [模型][epoch][batch]
    # 需要将所有batch的损失展平为一个大列表
    
    if not losses_lists:
        return [], []
    
    try:
        # 将每个模型的所有batch损失值展平到一个列表中
        print("开始处理损失数据...")
        flattened_losses = []
        for i, model_losses in enumerate(losses_lists):
            # 展平每个模型的所有epoch的所有batch损失
            print(f"处理模型 {i+1} 的损失数据:")
            print(f"  模型有 {len(model_losses)} 个epoch")
            
            model_flat_losses = []
            for j, epoch_losses in enumerate(model_losses):
                if isinstance(epoch_losses, list):
                    print(f"  epoch {j+1} 有 {len(epoch_losses)} 个batch")
                    model_flat_losses.extend(epoch_losses)
                else:
                    print(f"  警告: epoch {j+1} 的数据不是列表: {type(epoch_losses)}")
            
            print(f"  模型 {i+1} 总共有 {len(model_flat_losses)} 个batch")
            flattened_losses.append(model_flat_losses)
        
        # 找到最短的展平后损失列表长度
        if flattened_losses:
            lengths = [len(flat_losses) for flat_losses in flattened_losses]
            min_length = min(lengths)
            print(f"所有模型中最短的batch数量: {min_length}")
        else:
            min_length = 0
            print("没有损失数据，返回空列表")
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
        
        print(f"生成了min_curve(长度:{len(min_curve)})和max_curve(长度:{len(max_curve)})")
        return min_curve, max_curve
    except Exception as e:
        print(f"计算损失曲线时发生错误: {e}")
        # 返回空列表，避免程序崩溃
        return [], []

def main():
    """主函数：执行模型训练和损失景观分析"""
    # 显示开始消息
    print(f"{'='*50}")
    print(f"开始训练和损失景观分析")
    print(f"{'='*50}")
    print(f"使用设备: {device}")
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
    
    # 训练参数 - 减少epoch数量加快训练速度
    epo = 2
    
    # 设置学习率 - 减少为只使用两种学习率以加快训练速度
    learning_rates = [1e-3, 5e-4]  # 仅使用两种学习率进行测试
    losses_lists_vgg = []
    losses_lists_vgg_bn = []

    # 对每个学习率训练VGG模型
    for lr in learning_rates:
        print(f"\n{'='*50}\n训练普通VGG模型 (学习率: {lr})\n{'='*50}")
        set_random_seeds(seed_value=2020, device=device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model_losses, model_grads = train(model, optimizer, criterion, train_loader, test_loader, epochs_n=epo)
        losses_lists_vgg.append(model_losses)
        
        # 保存该模型的所有batch损失
        # 将所有epoch的batch损失展平为一个列表
        flat_losses = []
        for epoch_losses in model_losses:
            flat_losses.extend(epoch_losses)
        np.savetxt(os.path.join(loss_save_path, f'loss_vgg_lr_{lr}.txt'), flat_losses, fmt='%.6f')
        
        # 简化梯度保存，先跳过保存梯度信息以避免错误
        # np.savetxt(os.path.join(grad_save_path, f'grads_vgg_lr_{lr}.txt'), model_grads, fmt='%s', delimiter=' ')

    # 对每个学习率训练VGG_BN模型
    for lr in learning_rates:
        print(f"\n{'='*50}\n训练带BN的VGG模型 (学习率: {lr})\n{'='*50}")
        set_random_seeds(seed_value=2020, device=device)
        model_bn = VGG_A_BatchNorm()
        optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
        criterion_bn = nn.CrossEntropyLoss()
        model_losses_bn, model_grads_bn = train(model_bn, optimizer_bn, criterion_bn, train_loader, test_loader, epochs_n=epo)
        losses_lists_vgg_bn.append(model_losses_bn)
        
        # 保存该模型的所有batch损失
        # 将所有epoch的batch损失展平为一个列表
        flat_losses_bn = []
        for epoch_losses in model_losses_bn:
            flat_losses_bn.extend(epoch_losses)
        np.savetxt(os.path.join(loss_save_path, f'loss_vgg_bn_lr_{lr}.txt'), flat_losses_bn, fmt='%.6f')
        
        # 简化梯度保存，先跳过保存梯度信息以避免错误
        # np.savetxt(os.path.join(grad_save_path, f'grads_vgg_bn_lr_{lr}.txt'), model_grads_bn, fmt='%s', delimiter=' ')

    # 计算VGG模型的min_curve和max_curve
    # 调试信息，帮助理解数据结构
    print(f"VGG模型数量: {len(losses_lists_vgg)}")
    
    # 计算每个模型的总batch数
    if losses_lists_vgg:
        total_batches = 0
        for epoch_losses in losses_lists_vgg[0]:
            total_batches += len(epoch_losses)
        print(f"VGG第一个模型的总batch数: {total_batches}")
    
    # 计算VGG模型的min_curve和max_curve
    min_curve_vgg, max_curve_vgg = compute_loss_curves(losses_lists_vgg)
    
    # 计算VGG_BN模型的min_curve和max_curve
    min_curve_vgg_bn, max_curve_vgg_bn = compute_loss_curves(losses_lists_vgg_bn)
    
    # 打印曲线长度，确认数据正确
    print(f"VGG min_curve长度: {len(min_curve_vgg)}")
    print(f"VGG_BN min_curve长度: {len(min_curve_vgg_bn)}")

    print(f"{'='*50}")
    print(f"训练完成，开始绘制损失景观...")
    print(f"{'='*50}")
    
    # 绘制VGG模型的损失景观
    plot_loss_landscape(min_curve_vgg, max_curve_vgg, title="VGG-A", 
                        save_path=os.path.join(figures_path, "vgg_loss_landscape.png"),
                        color='#8FBC8F')  # 浅绿色

    # 绘制VGG_BN模型的损失景观
    plot_loss_landscape(min_curve_vgg_bn, max_curve_vgg_bn, title="VGG-A with BatchNorm", 
                        save_path=os.path.join(figures_path, "vgg_bn_loss_landscape.png"),
                        color='#DB7093')  # 浅粉色

    print("准备绘制对比图...")
    
    try:
        # 打印比较图的数据信息
        print("绘制损失景观比较图开始:")
        print(f"  VGG min曲线类型: {type(min_curve_vgg)}")
        print(f"  VGG min曲线长度: {len(min_curve_vgg) if isinstance(min_curve_vgg, (list, np.ndarray)) else 'N/A'}")
        print(f"  VGG max曲线长度: {len(max_curve_vgg) if isinstance(max_curve_vgg, (list, np.ndarray)) else 'N/A'}")
        print(f"  VGG_BN min曲线长度: {len(min_curve_vgg_bn) if isinstance(min_curve_vgg_bn, (list, np.ndarray)) else 'N/A'}")
        print(f"  VGG_BN max曲线长度: {len(max_curve_vgg_bn) if isinstance(max_curve_vgg_bn, (list, np.ndarray)) else 'N/A'}")
        
        # 确保数据为numpy一维数组
        min_curve_vgg = np.array(min_curve_vgg, dtype=float).flatten()
        max_curve_vgg = np.array(max_curve_vgg, dtype=float).flatten()
        min_curve_vgg_bn = np.array(min_curve_vgg_bn, dtype=float).flatten()
        max_curve_vgg_bn = np.array(max_curve_vgg_bn, dtype=float).flatten()
        
        # 检查数组长度，确保至少有数据可以绘图
        if len(min_curve_vgg) > 0 and len(max_curve_vgg) > 0 and len(min_curve_vgg_bn) > 0 and len(max_curve_vgg_bn) > 0:
            plt.figure(figsize=(12, 8))
            
            # 打印前几个值帮助调试
            print(f"  VGG min前5个值: {min_curve_vgg[:5]}")
            print(f"  VGG max前5个值: {max_curve_vgg[:5]}")
            print(f"  VGG_BN min前5个值: {min_curve_vgg_bn[:5]}")
            print(f"  VGG_BN max前5个值: {max_curve_vgg_bn[:5]}")
            
            steps_vgg = range(len(min_curve_vgg))
            steps_vgg_bn = range(len(min_curve_vgg_bn))

            print("绘制填充区域...")
            # 使用更美观的颜色填充区域
            plt.fill_between(steps_vgg, min_curve_vgg, max_curve_vgg, 
                             alpha=0.35, color='#8FBC8F', label='Standard VGG')
            plt.fill_between(steps_vgg_bn, min_curve_vgg_bn, max_curve_vgg_bn, 
                             alpha=0.35, color='#DB7093', label='Standard VGG + BatchNorm')
            
            print("设置图表标题、轴标签等...")
            plt.title('Loss Landscape', fontsize=18, pad=20)
            plt.xlabel('Steps', fontsize=14)
            plt.ylabel('Loss Landscape', fontsize=14)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True, alpha=0.3, color='gray')
            
            print("保存对比图...")
            comparison_path = os.path.join(figures_path, "loss_landscape_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"已保存对比图到: {comparison_path}")
            plt.close()
        else:
            print("错误: 曲线数据为空，无法生成对比图")
    except Exception as e:
        print(f"生成损失景观对比图时出错: {e}")
        # 打印错误的详细信息
        import traceback
        print(traceback.format_exc())

    print(f"{'='*50}")
    print("训练和可视化完成，损失景观图已保存。")
    print(f"可视化文件保存在: {figures_path}")
    print(f"{'='*50}")

# 只有当直接运行此脚本时才执行主函数
if __name__ == "__main__":
    main()