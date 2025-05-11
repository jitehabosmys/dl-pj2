import matplotlib.pyplot as plt
import os
import numpy as np

# 设置matplotlib样式，确保可重现性
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(42)  # 确保颜色等随机选择的一致性

def visualize_results(train_losses, train_accs, model_name, save_dir="results/images"):
    """可视化训练结果并保存图像"""
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失变化
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率变化
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'r-')
    plt.title(f'{model_name} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 正确的保存路径（替换反斜杠为正斜杠，确保跨平台兼容）
    save_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    
    # 保存图像
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    
    plt.close()  # 关闭图形，避免显示问题

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