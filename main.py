import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# 导入自定义模块
from data.loaders import get_cifar_loader
from models.cifar_net import BasicCNN, CNNWithBatchNorm, CNNWithDropout, ResNet
from utils.trainer import train, evaluate, set_seed
from utils.visualization import visualize_results, compare_models
from utils.model_utils import count_parameters, save_model, get_optimizer

def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 确保结果目录存在
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)
    
    # 加载数据
    train_loader = get_cifar_loader(root='./data', train=True, batch_size=128, download=True)
    test_loader = get_cifar_loader(root='./data', train=False, batch_size=128, download=True)
    
    # 模型配置
    models = {
        'BasicCNN': BasicCNN(),
        'CNNWithBatchNorm': CNNWithBatchNorm(),
        'CNNWithDropout': CNNWithDropout(dropout_rate=0.25),
        'ResNet': ResNet(num_blocks=2)
    }
    
    # 训练参数
    epochs = 6
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    
    # 存储结果
    results = {}
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # 将模型移至设备
        model = model.to(device)
        
        # 选择优化器
        optimizer = get_optimizer(model, opt_name='adam', lr=learning_rate)
        
        # 计算模型参数量
        param_count = count_parameters(model)
        print(f"Model has {param_count:.2f}M parameters")
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练模型
        train_losses, train_accs = train(model, train_loader, criterion, optimizer, device, epochs=epochs)
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 可视化训练结果
        visualize_results(train_losses, train_accs, model_name, save_dir="results/images")
        
        # 保存模型
        save_model(model, model_name, save_dir="results/models")
        
        # 存储结果
        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'param_count': param_count,
            'training_time': training_time
        }
        
        print(f"Training time: {training_time:.2f} seconds")
    
    # 比较模型性能
    compare_models(results, save_dir="results/images")
    
    # 输出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])[0]
    print(f"\nBest model: {best_model} with test accuracy {results[best_model]['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
