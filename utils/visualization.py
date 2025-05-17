import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn

# Set matplotlib style for reproducibility
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)  # Ensure consistency in random selections

def visualize_results(train_losses, train_accs, model_name, save_dir="results/images"):
    """Visualize training results and save images"""
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss change
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot accuracy change
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'r-')
    plt.title(f'{model_name} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_training_results.png"), dpi=200)
    plt.close()

def compare_models(results, save_dir="results/images", comparison_name="model_comparison"):
    """比较不同模型的性能"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = list(results.keys())
    test_accs = [results[name]['test_acc'] for name in model_names]
    param_counts = [results[name]['param_count'] for name in model_names]
    train_times = [results[name]['training_time'] for name in model_names]
    
    # 绘制准确率比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, test_accs, color='skyblue')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha='center'
        )
    
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f'{comparison_name}_accuracy.png')
    plt.savefig(acc_path)
    print(f"Accuracy comparison saved to {acc_path}")
    plt.close()
    
    # 绘制参数量比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, param_counts, color='salmon')
    plt.title('Model Parameter Count Comparison')
    plt.xlabel('Model')
    plt.ylabel('Number of Parameters (M)')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, param_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            f"{value:.2f}M",
            ha='center'
        )
    
    plt.tight_layout()
    param_path = os.path.join(save_dir, f'{comparison_name}_params.png')
    plt.savefig(param_path)
    print(f"Parameter comparison saved to {param_path}")
    plt.close()
    
    # 绘制训练时间比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, train_times, color='lightgreen')
    plt.title('Training Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, train_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(train_times) * 0.05,
            f"{value:.1f}s",
            ha='center'
        )
    
    plt.tight_layout()
    time_path = os.path.join(save_dir, f'{comparison_name}_time.png')
    plt.savefig(time_path)
    print(f"Training time comparison saved to {time_path}")
    plt.close()

def visualize_conv_filters(model, save_dir="results/images", model_name="model"):
    """Visualize convolutional filters in the model"""
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all convolutional layers
    conv_layers = []
    layer_names = []
    
    # Process model to handle different structures (Sequential, ModuleList, etc.)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
            layer_names.append(name)
    
    print(f"Found {len(conv_layers)} convolutional layers to visualize")
    
    # Visualize each convolutional layer
    for i, (layer, name) in enumerate(zip(conv_layers, layer_names)):
        # Ensure name is valid for filename
        safe_name = name.replace('.', '_').replace('/', '_')
        visualize_layer_filters(layer, i, name, 
                              save_path=os.path.join(save_dir, f"{model_name}_layer_{i}_{safe_name}.png"))

def visualize_layer_filters(layer, layer_idx, layer_name, save_path):
    """Visualize filters of a single convolutional layer"""
    # Get layer weights (filters)
    weights = layer.weight.data.cpu().numpy()
    
    # Get filter dimensions
    n_filters, n_channels, height, width = weights.shape
    
    print(f"Layer {layer_idx}: {layer_name}, Shape: {weights.shape}")
    
    # Calculate number of columns and rows for subplot
    n_cols = min(8, n_filters)  # Maximum 8 columns for better display
    n_rows = int(np.ceil(n_filters / n_cols))
    
    # Create figure
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    # Set overall title
    plt.suptitle(f"Layer {layer_idx}: {layer_name} - Filters", fontsize=16)
    
    # Plot each filter
    for j in range(n_filters):
        filter_weights = weights[j]
        
        # For the first layer (RGB input), display as color
        if n_channels == 3 and layer_idx == 0:
            # Normalize to [0, 1] for display
            filter_rgb = process_filter_for_display(filter_weights)
            
            plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(filter_rgb)
            plt.title(f"Filter {j+1}")
            plt.axis('off')
        else:
            # For deeper layers with multiple channels, take the average across channels
            filter_avg = np.mean(filter_weights, axis=0)
            
            # Normalize for display
            filter_avg = (filter_avg - filter_avg.min()) / (filter_avg.max() - filter_avg.min() + 1e-8)
            
            plt.subplot(n_rows, n_cols, j + 1)
            # Use grayscale colormap for deeper layers
            plt.imshow(filter_avg, cmap='gray')
            plt.title(f"Filter {j+1}")
            plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(save_path, dpi=200)
    plt.close()

def process_filter_for_display(filter_weights):
    """Process a 3-channel filter for RGB display"""
    # Transpose to (H, W, C) format for imshow
    filter_rgb = np.transpose(filter_weights, (1, 2, 0))
    
    # Normalize each channel independently to [0, 1]
    for c in range(3):
        channel = filter_rgb[:, :, c]
        min_val = channel.min()
        max_val = channel.max()
        filter_rgb[:, :, c] = (channel - min_val) / (max_val - min_val + 1e-8)
    
    return filter_rgb

def visualize_first_layer_rgb(weights, layer_name, save_dir, model_name):
    """使用RGB通道可视化第一层卷积核"""
    n_filters = weights.shape[0]
    
    # 计算子图的行列数
    n_cols = min(8, n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    fig.suptitle('第一层卷积核的RGB可视化', fontsize=16)
    
    # 将轴展平方便索引
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 为每个滤波器绘制RGB图像
    for i in range(n_rows * n_cols):
        if i < n_filters:
            # 将卷积核权重转换为RGB图像格式
            # 形状从(channels, height, width)转为(height, width, channels)
            filter_vis = np.transpose(weights[i], (1, 2, 0))
            
            # 归一化到[0,1]范围用于显示
            filter_vis = (filter_vis - filter_vis.min()) / (filter_vis.max() - filter_vis.min() + 1e-5)
            
            # 绘制图像
            axes[i].imshow(filter_vis)
            axes[i].set_title(f'Filter #{i+1}')
        
        # 关闭刻度
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间
    
    # 安全的文件名
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(save_dir, f'{model_name}_filters_rgb_{safe_layer_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存第一层卷积核的RGB可视化") 