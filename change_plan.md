# 关于当前项目的更改

## 问题
我感觉我们现在的项目结构比较繁杂，功能实现也需要重新调整。比如有以下几点我感觉不满意的地方。

### 模型定义的清晰性
models/cifar_net.py 中定义了四个不同的模型，但更清晰地，我们应该把每个模型独立用一个文件定义。

### 实现模型的必要性
- 在 @project_2_2025.md 中说到：

Yes, you are totally free to use any component in torch, pytorch or other deep learning toolbox. Then how do we score your project in this task? We care about:

1. (12%) The classification performance of your network, in terms of total parameters of network, network structure, and training speed. For the projects of similar results, we will check the total parameters, and network structure, or whether any new optimization algorithms are used to train the network.

2. Any insightful results and interesting visualization of the learned model, and training process, or anything like this.

3. You can report the results from more than one network; but if you directly utilize the public available models without anything changes, the scores of your projects would be slightly penalized.

既然我们的评分是基于模型的效率和准确率，那么我们应该选择一个已经被实践证明在较小参数量下取得好的结果的经典模型结构，作为实验的基础。比如说：RegNetX_200MF（参数约2.7M），ResNet18（参数约11.2M）等。
当前models/cifar_net.py中的四个模型仅能达到80%左右的正确率，效果不佳，再次基础上优化也很难提升。

### 实验的设置
当前的脚本文件夹下的定义令人不安。@compare.py @experiment.py 的实现暂时没有必要，我们只需保留 @train.py @test.py 两个最关键的脚本即可。回忆任务要求中关于实验的部分：

2. Your network will have all the following components: (16%)

- (a) Fully-Connected layer  
- (b) 2D convolutional layer  
- (c) 2D pooling layer  
- (d) Activations  

3. Your network may have some of (at least one) the following components: (8%)

- (a) Batch-Norm layer  
- (b) Drop out  
- (c) Residual Connection  
- (d) Others  

4. To optimize your network, you will try all the following strategies: (8%)

- (a) Try different number of neurons/filters  
- (b) Try different loss functions (with different regularization)  
- (c) Try different activations  

5. To optimize your network, you may select one of the following strategies: (8%)

- (a) Try different optimizers using `torch.optim`  
- (b) Implement an optimizer for a network including 2(a)-(d), and use `torch.optim` to optimize your full model  
- (c) Implement an optimizer for your full model  

其中第2和3两点我们可以通过选择具有如此结构的模型来完成，而不用像之前那样定义四个模型（最基础、加bn，加dropout，加残差连接）。而其他实验，我们可以选择在train.py中选择进行相应的参数解析来实现。

## 参考
@ pytorch-cifar-master 这是我从github上找到的一个代码，使用多种经典模型对cifar10做了实验，准确率都比较高。我们可以借鉴其中的模型来选择实现。
使用设备: cuda
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|████████████████████████████████████████| 170M/170M [00:16<00:00, 10.2MB/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
训练集大小: 45000, 验证集大小: 5000
Files already downloaded and verified
模型有 11.17M 参数
使用CosineAnnealingLR学习率调度器，T_max=60

==================================================
训练 ResNet18
==================================================
Epoch 1/60: 100%|████| 352/352 [00:26<00:00, 13.11it/s, loss=1.8834, acc=31.91%]
Validating Epoch 1: 100%|█| 40/40 [00:01<00:00, 29.62it/s, loss=1.5351, acc=43.1
Epoch 1/60, Loss: 1.8834, Acc: 31.91%, Val Loss: 1.5351, Val Acc: 43.16%, LR: 0.100000
验证损失从 inf 改善到 1.5351，保存模型状态
Epoch 2/60: 100%|████| 352/352 [00:25<00:00, 13.67it/s, loss=1.5950, acc=40.76%]
Validating Epoch 2: 100%|█| 40/40 [00:01<00:00, 31.38it/s, loss=1.5111, acc=44.7
Epoch 2/60, Loss: 1.5950, Acc: 40.76%, Val Loss: 1.5111, Val Acc: 44.72%, LR: 0.099931
验证损失从 1.5351 改善到 1.5111，保存模型状态
Epoch 3/60: 100%|████| 352/352 [00:25<00:00, 13.78it/s, loss=1.3765, acc=49.65%]
Validating Epoch 3: 100%|█| 40/40 [00:01<00:00, 30.67it/s, loss=1.3313, acc=52.7
Epoch 3/60, Loss: 1.3765, Acc: 49.65%, Val Loss: 1.3313, Val Acc: 52.70%, LR: 0.099726
验证损失从 1.5111 改善到 1.3313，保存模型状态
Epoch 4/60: 100%|████| 352/352 [00:25<00:00, 13.60it/s, loss=1.2546, acc=54.44%]
Validating Epoch 4: 100%|█| 40/40 [00:01<00:00, 31.03it/s, loss=1.2039, acc=57.1
Epoch 4/60, Loss: 1.2546, Acc: 54.44%, Val Loss: 1.2039, Val Acc: 57.14%, LR: 0.099384
验证损失从 1.3313 改善到 1.2039，保存模型状态
Epoch 5/60: 100%|████| 352/352 [00:25<00:00, 13.78it/s, loss=1.1585, acc=58.40%]
Validating Epoch 5: 100%|█| 40/40 [00:01<00:00, 22.27it/s, loss=1.2010, acc=57.2
Epoch 5/60, Loss: 1.1585, Acc: 58.40%, Val Loss: 1.2010, Val Acc: 57.28%, LR: 0.098907
验证损失从 1.2039 改善到 1.2010，保存模型状态
Epoch 6/60: 100%|████| 352/352 [00:25<00:00, 13.78it/s, loss=1.1056, acc=60.55%]
Validating Epoch 6: 100%|█| 40/40 [00:01<00:00, 32.36it/s, loss=1.1400, acc=59.5
Epoch 6/60, Loss: 1.1056, Acc: 60.55%, Val Loss: 1.1400, Val Acc: 59.54%, LR: 0.098296
验证损失从 1.2010 改善到 1.1400，保存模型状态
Epoch 7/60: 100%|████| 352/352 [00:25<00:00, 13.66it/s, loss=1.0481, acc=62.60%]
Validating Epoch 7: 100%|█| 40/40 [00:01<00:00, 32.51it/s, loss=1.0288, acc=63.8
Epoch 7/60, Loss: 1.0481, Acc: 62.60%, Val Loss: 1.0288, Val Acc: 63.88%, LR: 0.097553
验证损失从 1.1400 改善到 1.0288，保存模型状态
Epoch 8/60: 100%|████| 352/352 [00:25<00:00, 13.78it/s, loss=1.0120, acc=63.91%]
Validating Epoch 8: 100%|█| 40/40 [00:01<00:00, 30.62it/s, loss=0.9843, acc=64.8
Epoch 8/60, Loss: 1.0120, Acc: 63.91%, Val Loss: 0.9843, Val Acc: 64.88%, LR: 0.096679
验证损失从 1.0288 改善到 0.9843，保存模型状态
Epoch 9/60: 100%|████| 352/352 [00:25<00:00, 13.64it/s, loss=0.9816, acc=65.32%]
Validating Epoch 9: 100%|█| 40/40 [00:01<00:00, 33.11it/s, loss=1.0037, acc=63.8
Epoch 9/60, Loss: 0.9816, Acc: 65.32%, Val Loss: 1.0037, Val Acc: 63.84%, LR: 0.095677
EarlyStopping 计数器: 1/20
Epoch 10/60: 100%|███| 352/352 [00:25<00:00, 13.77it/s, loss=0.9525, acc=66.29%]
Validating Epoch 10: 100%|█| 40/40 [00:01<00:00, 33.83it/s, loss=0.9825, acc=64.
Epoch 10/60, Loss: 0.9525, Acc: 66.29%, Val Loss: 0.9825, Val Acc: 64.40%, LR: 0.094550
验证损失从 0.9843 改善到 0.9825，保存模型状态
Epoch 11/60: 100%|███| 352/352 [00:25<00:00, 13.65it/s, loss=0.9320, acc=66.95%]
Validating Epoch 11: 100%|█| 40/40 [00:01<00:00, 31.62it/s, loss=0.9548, acc=66.
Epoch 11/60, Loss: 0.9320, Acc: 66.95%, Val Loss: 0.9548, Val Acc: 66.30%, LR: 0.093301
验证损失从 0.9825 改善到 0.9548，保存模型状态
Epoch 12/60: 100%|███| 352/352 [00:25<00:00, 13.76it/s, loss=0.9010, acc=68.37%]
Validating Epoch 12: 100%|█| 40/40 [00:01<00:00, 31.10it/s, loss=0.8579, acc=69.
Epoch 12/60, Loss: 0.9010, Acc: 68.37%, Val Loss: 0.8579, Val Acc: 69.82%, LR: 0.091934
验证损失从 0.9548 改善到 0.8579，保存模型状态
Epoch 13/60: 100%|███| 352/352 [00:25<00:00, 13.65it/s, loss=0.8790, acc=69.20%]
Validating Epoch 13: 100%|█| 40/40 [00:01<00:00, 32.52it/s, loss=0.8515, acc=70.
Epoch 13/60, Loss: 0.8790, Acc: 69.20%, Val Loss: 0.8515, Val Acc: 70.60%, LR: 0.090451
验证损失从 0.8579 改善到 0.8515，保存模型状态
Epoch 14/60: 100%|███| 352/352 [00:25<00:00, 13.76it/s, loss=0.8577, acc=69.60%]
Validating Epoch 14: 100%|█| 40/40 [00:01<00:00, 28.67it/s, loss=0.7891, acc=72.
Epoch 14/60, Loss: 0.8577, Acc: 69.60%, Val Loss: 0.7891, Val Acc: 72.16%, LR: 0.088857
验证损失从 0.8515 改善到 0.7891，保存模型状态
Epoch 15/60: 100%|███| 352/352 [00:25<00:00, 13.64it/s, loss=0.8344, acc=70.61%]
Validating Epoch 15: 100%|█| 40/40 [00:01<00:00, 31.86it/s, loss=0.8055, acc=72.
Epoch 15/60, Loss: 0.8344, Acc: 70.61%, Val Loss: 0.8055, Val Acc: 72.30%, LR: 0.087157
EarlyStopping 计数器: 1/20
Epoch 16/60: 100%|███| 352/352 [00:25<00:00, 13.76it/s, loss=0.8051, acc=71.50%]
Validating Epoch 16: 100%|█| 40/40 [00:01<00:00, 30.16it/s, loss=0.8451, acc=70.
Epoch 16/60, Loss: 0.8051, Acc: 71.50%, Val Loss: 0.8451, Val Acc: 70.56%, LR: 0.085355
EarlyStopping 计数器: 2/20
Epoch 17/60: 100%|███| 352/352 [00:25<00:00, 13.65it/s, loss=0.7882, acc=72.13%]
Validating Epoch 17: 100%|█| 40/40 [00:01<00:00, 31.79it/s, loss=0.8256, acc=71.
Epoch 17/60, Loss: 0.7882, Acc: 72.13%, Val Loss: 0.8256, Val Acc: 71.46%, LR: 0.083457
EarlyStopping 计数器: 3/20
Epoch 18/60: 100%|███| 352/352 [00:25<00:00, 13.77it/s, loss=0.7618, acc=73.17%]
Validating Epoch 18: 100%|█| 40/40 [00:01<00:00, 31.58it/s, loss=0.9598, acc=67.
Epoch 18/60, Loss: 0.7618, Acc: 73.17%, Val Loss: 0.9598, Val Acc: 67.76%, LR: 0.081466
EarlyStopping 计数器: 4/20
Epoch 19/60: 100%|███| 352/352 [00:25<00:00, 13.65it/s, loss=0.7338, acc=74.50%]
Validating Epoch 19: 100%|█| 40/40 [00:01<00:00, 29.98it/s, loss=0.7216, acc=74.
Epoch 19/60, Loss: 0.7338, Acc: 74.50%, Val Loss: 0.7216, Val Acc: 74.94%, LR: 0.079389
验证损失从 0.7891 改善到 0.7216，保存模型状态
Epoch 20/60: 100%|███| 352/352 [00:25<00:00, 13.74it/s, loss=0.7128, acc=75.04%]
Validating Epoch 20: 100%|█| 40/40 [00:01<00:00, 31.27it/s, loss=0.7123, acc=75.
Epoch 20/60, Loss: 0.7128, Acc: 75.04%, Val Loss: 0.7123, Val Acc: 75.12%, LR: 0.077232
验证损失从 0.7216 改善到 0.7123，保存模型状态
Epoch 21/60: 100%|███| 352/352 [00:25<00:00, 13.64it/s, loss=0.6870, acc=76.19%]
Validating Epoch 21: 100%|█| 40/40 [00:01<00:00, 30.62it/s, loss=0.6298, acc=77.
Epoch 21/60, Loss: 0.6870, Acc: 76.19%, Val Loss: 0.6298, Val Acc: 77.76%, LR: 0.075000
验证损失从 0.7123 改善到 0.6298，保存模型状态
Epoch 22/60: 100%|███| 352/352 [00:25<00:00, 13.73it/s, loss=0.6812, acc=76.21%]
Validating Epoch 22: 100%|█| 40/40 [00:01<00:00, 32.83it/s, loss=0.6599, acc=76.
Epoch 22/60, Loss: 0.6812, Acc: 76.21%, Val Loss: 0.6599, Val Acc: 76.78%, LR: 0.072700
EarlyStopping 计数器: 1/20
Epoch 23/60: 100%|███| 352/352 [00:25<00:00, 13.64it/s, loss=0.6560, acc=77.28%]
Validating Epoch 23: 100%|█| 40/40 [00:01<00:00, 32.44it/s, loss=0.6575, acc=77.
Epoch 23/60, Loss: 0.6560, Acc: 77.28%, Val Loss: 0.6575, Val Acc: 77.00%, LR: 0.070337
EarlyStopping 计数器: 2/20
Epoch 24/60: 100%|███| 352/352 [00:25<00:00, 13.74it/s, loss=0.6478, acc=77.42%]
Validating Epoch 24: 100%|█| 40/40 [00:01<00:00, 26.42it/s, loss=0.6198, acc=78.
Epoch 24/60, Loss: 0.6478, Acc: 77.42%, Val Loss: 0.6198, Val Acc: 78.68%, LR: 0.067918
验证损失从 0.6298 改善到 0.6198，保存模型状态
Epoch 25/60: 100%|███| 352/352 [00:25<00:00, 13.64it/s, loss=0.6226, acc=78.37%]
Validating Epoch 25: 100%|█| 40/40 [00:01<00:00, 30.80it/s, loss=0.6436, acc=77.
Epoch 25/60, Loss: 0.6226, Acc: 78.37%, Val Loss: 0.6436, Val Acc: 77.44%, LR: 0.065451
EarlyStopping 计数器: 1/20
Epoch 26/60: 100%|███| 352/352 [00:25<00:00, 13.74it/s, loss=0.6125, acc=78.54%]
Validating Epoch 26: 100%|█| 40/40 [00:01<00:00, 31.93it/s, loss=0.7211, acc=76.
Epoch 26/60, Loss: 0.6125, Acc: 78.54%, Val Loss: 0.7211, Val Acc: 76.22%, LR: 0.062941
EarlyStopping 计数器: 2/20
Epoch 27/60: 100%|███| 352/352 [00:25<00:00, 13.64it/s, loss=0.5939, acc=79.44%]
Validating Epoch 27: 100%|█| 40/40 [00:01<00:00, 31.46it/s, loss=0.5857, acc=79.
Epoch 27/60, Loss: 0.5939, Acc: 79.44%, Val Loss: 0.5857, Val Acc: 79.86%, LR: 0.060396
验证损失从 0.6198 改善到 0.5857，保存模型状态
Epoch 28/60: 100%|███| 352/352 [00:25<00:00, 13.73it/s, loss=0.5823, acc=79.78%]
Validating Epoch 28: 100%|█| 40/40 [00:01<00:00, 30.44it/s, loss=0.6218, acc=78.
Epoch 28/60, Loss: 0.5823, Acc: 79.78%, Val Loss: 0.6218, Val Acc: 78.68%, LR: 0.057822
EarlyStopping 计数器: 1/20
Epoch 29/60: 100%|███| 352/352 [00:25<00:00, 13.62it/s, loss=0.5666, acc=80.47%]
Validating Epoch 29: 100%|█| 40/40 [00:01<00:00, 31.54it/s, loss=0.6178, acc=79.
Epoch 29/60, Loss: 0.5666, Acc: 80.47%, Val Loss: 0.6178, Val Acc: 79.04%, LR: 0.055226
EarlyStopping 计数器: 2/20
Epoch 30/60: 100%|███| 352/352 [00:25<00:00, 13.71it/s, loss=0.5650, acc=80.62%]
Validating Epoch 30: 100%|█| 40/40 [00:01<00:00, 23.48it/s, loss=0.5621, acc=80.
Epoch 30/60, Loss: 0.5650, Acc: 80.62%, Val Loss: 0.5621, Val Acc: 80.24%, LR: 0.052617
验证损失从 0.5857 改善到 0.5621，保存模型状态
Epoch 31/60: 100%|███| 352/352 [00:25<00:00, 13.72it/s, loss=0.5446, acc=81.16%]
Validating Epoch 31: 100%|█| 40/40 [00:01<00:00, 30.17it/s, loss=0.5754, acc=80.
Epoch 31/60, Loss: 0.5446, Acc: 81.16%, Val Loss: 0.5754, Val Acc: 80.50%, LR: 0.050000
EarlyStopping 计数器: 1/20
Epoch 32/60: 100%|███| 352/352 [00:25<00:00, 13.61it/s, loss=0.5379, acc=81.45%]
Validating Epoch 32: 100%|█| 40/40 [00:01<00:00, 32.43it/s, loss=0.6053, acc=78.
Epoch 32/60, Loss: 0.5379, Acc: 81.45%, Val Loss: 0.6053, Val Acc: 78.74%, LR: 0.047383
EarlyStopping 计数器: 2/20
Epoch 33/60: 100%|███| 352/352 [00:25<00:00, 13.70it/s, loss=0.5243, acc=81.68%]
Validating Epoch 33: 100%|█| 40/40 [00:01<00:00, 31.54it/s, loss=0.5370, acc=81.
Epoch 33/60, Loss: 0.5243, Acc: 81.68%, Val Loss: 0.5370, Val Acc: 81.80%, LR: 0.044774
验证损失从 0.5621 改善到 0.5370，保存模型状态
Epoch 34/60: 100%|███| 352/352 [00:25<00:00, 13.60it/s, loss=0.5124, acc=82.18%]
Validating Epoch 34: 100%|█| 40/40 [00:01<00:00, 31.83it/s, loss=0.5223, acc=82.
Epoch 34/60, Loss: 0.5124, Acc: 82.18%, Val Loss: 0.5223, Val Acc: 82.00%, LR: 0.042178
验证损失从 0.5370 改善到 0.5223，保存模型状态
Epoch 35/60: 100%|███| 352/352 [00:25<00:00, 13.70it/s, loss=0.4942, acc=82.90%]
Validating Epoch 35: 100%|█| 40/40 [00:01<00:00, 30.48it/s, loss=0.5254, acc=81.
Epoch 35/60, Loss: 0.4942, Acc: 82.90%, Val Loss: 0.5254, Val Acc: 81.56%, LR: 0.039604
EarlyStopping 计数器: 1/20
Epoch 36/60: 100%|███| 352/352 [00:25<00:00, 13.60it/s, loss=0.4794, acc=83.48%]
Validating Epoch 36: 100%|█| 40/40 [00:01<00:00, 32.99it/s, loss=0.5061, acc=82.
Epoch 36/60, Loss: 0.4794, Acc: 83.48%, Val Loss: 0.5061, Val Acc: 82.52%, LR: 0.037059
验证损失从 0.5223 改善到 0.5061，保存模型状态
Epoch 37/60: 100%|███| 352/352 [00:25<00:00, 13.71it/s, loss=0.4719, acc=83.54%]
Validating Epoch 37: 100%|█| 40/40 [00:01<00:00, 30.41it/s, loss=0.4952, acc=82.
Epoch 37/60, Loss: 0.4719, Acc: 83.54%, Val Loss: 0.4952, Val Acc: 82.50%, LR: 0.034549
验证损失从 0.5061 改善到 0.4952，保存模型状态
Epoch 38/60: 100%|███| 352/352 [00:25<00:00, 13.57it/s, loss=0.4567, acc=84.13%]
Validating Epoch 38: 100%|█| 40/40 [00:01<00:00, 31.32it/s, loss=0.5109, acc=82.
Epoch 38/60, Loss: 0.4567, Acc: 84.13%, Val Loss: 0.5109, Val Acc: 82.58%, LR: 0.032082
EarlyStopping 计数器: 1/20
Epoch 39/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.4486, acc=84.53%]
Validating Epoch 39: 100%|█| 40/40 [00:01<00:00, 29.80it/s, loss=0.5109, acc=82.
Epoch 39/60, Loss: 0.4486, Acc: 84.53%, Val Loss: 0.5109, Val Acc: 82.98%, LR: 0.029663
EarlyStopping 计数器: 2/20
Epoch 40/60: 100%|███| 352/352 [00:25<00:00, 13.58it/s, loss=0.4345, acc=84.79%]
Validating Epoch 40: 100%|█| 40/40 [00:01<00:00, 32.78it/s, loss=0.4774, acc=83.
Epoch 40/60, Loss: 0.4345, Acc: 84.79%, Val Loss: 0.4774, Val Acc: 83.22%, LR: 0.027300
验证损失从 0.4952 改善到 0.4774，保存模型状态
Epoch 41/60: 100%|███| 352/352 [00:25<00:00, 13.70it/s, loss=0.4186, acc=85.42%]
Validating Epoch 41: 100%|█| 40/40 [00:01<00:00, 32.61it/s, loss=0.4654, acc=83.
Epoch 41/60, Loss: 0.4186, Acc: 85.42%, Val Loss: 0.4654, Val Acc: 83.98%, LR: 0.025000
验证损失从 0.4774 改善到 0.4654，保存模型状态
Epoch 42/60: 100%|███| 352/352 [00:25<00:00, 13.59it/s, loss=0.4147, acc=85.57%]
Validating Epoch 42: 100%|█| 40/40 [00:01<00:00, 32.47it/s, loss=0.4710, acc=84.
Epoch 42/60, Loss: 0.4147, Acc: 85.57%, Val Loss: 0.4710, Val Acc: 84.06%, LR: 0.022768
EarlyStopping 计数器: 1/20
Epoch 43/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.3964, acc=86.30%]
Validating Epoch 43: 100%|█| 40/40 [00:01<00:00, 30.08it/s, loss=0.4325, acc=84.
Epoch 43/60, Loss: 0.3964, Acc: 86.30%, Val Loss: 0.4325, Val Acc: 84.96%, LR: 0.020611
验证损失从 0.4654 改善到 0.4325，保存模型状态
Epoch 44/60: 100%|███| 352/352 [00:25<00:00, 13.60it/s, loss=0.3817, acc=86.77%]
Validating Epoch 44: 100%|█| 40/40 [00:01<00:00, 33.30it/s, loss=0.4459, acc=84.
Epoch 44/60, Loss: 0.3817, Acc: 86.77%, Val Loss: 0.4459, Val Acc: 84.54%, LR: 0.018534
EarlyStopping 计数器: 1/20
Epoch 45/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.3729, acc=87.15%]
Validating Epoch 45: 100%|█| 40/40 [00:01<00:00, 27.91it/s, loss=0.4275, acc=85.
Epoch 45/60, Loss: 0.3729, Acc: 87.15%, Val Loss: 0.4275, Val Acc: 85.22%, LR: 0.016543
验证损失从 0.4325 改善到 0.4275，保存模型状态
Epoch 46/60: 100%|███| 352/352 [00:25<00:00, 13.68it/s, loss=0.3596, acc=87.56%]
Validating Epoch 46: 100%|█| 40/40 [00:01<00:00, 29.41it/s, loss=0.4282, acc=85.
Epoch 46/60, Loss: 0.3596, Acc: 87.56%, Val Loss: 0.4282, Val Acc: 85.26%, LR: 0.014645
EarlyStopping 计数器: 1/20
Epoch 47/60: 100%|███| 352/352 [00:25<00:00, 13.59it/s, loss=0.3482, acc=87.94%]
Validating Epoch 47: 100%|█| 40/40 [00:01<00:00, 30.10it/s, loss=0.4249, acc=85.
Epoch 47/60, Loss: 0.3482, Acc: 87.94%, Val Loss: 0.4249, Val Acc: 85.48%, LR: 0.012843
验证损失从 0.4275 改善到 0.4249，保存模型状态
Epoch 48/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.3403, acc=87.89%]
Validating Epoch 48: 100%|█| 40/40 [00:01<00:00, 31.03it/s, loss=0.4015, acc=86.
Epoch 48/60, Loss: 0.3403, Acc: 87.89%, Val Loss: 0.4015, Val Acc: 86.46%, LR: 0.011143
验证损失从 0.4249 改善到 0.4015，保存模型状态
Epoch 49/60: 100%|███| 352/352 [00:25<00:00, 13.57it/s, loss=0.3185, acc=88.78%]
Validating Epoch 49: 100%|█| 40/40 [00:01<00:00, 32.37it/s, loss=0.4106, acc=85.
Epoch 49/60, Loss: 0.3185, Acc: 88.78%, Val Loss: 0.4106, Val Acc: 85.76%, LR: 0.009549
EarlyStopping 计数器: 1/20
Epoch 50/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.3106, acc=89.27%]
Validating Epoch 50: 100%|█| 40/40 [00:01<00:00, 30.25it/s, loss=0.3927, acc=86.
Epoch 50/60, Loss: 0.3106, Acc: 89.27%, Val Loss: 0.3927, Val Acc: 86.22%, LR: 0.008066
验证损失从 0.4015 改善到 0.3927，保存模型状态
Epoch 51/60: 100%|███| 352/352 [00:25<00:00, 13.67it/s, loss=0.2981, acc=89.64%]
Validating Epoch 51: 100%|█| 40/40 [00:01<00:00, 24.11it/s, loss=0.3985, acc=86.
Epoch 51/60, Loss: 0.2981, Acc: 89.64%, Val Loss: 0.3985, Val Acc: 86.42%, LR: 0.006699
EarlyStopping 计数器: 1/20
Epoch 52/60: 100%|███| 352/352 [00:25<00:00, 13.68it/s, loss=0.2904, acc=89.91%]
Validating Epoch 52: 100%|█| 40/40 [00:01<00:00, 32.15it/s, loss=0.3911, acc=86.
Epoch 52/60, Loss: 0.2904, Acc: 89.91%, Val Loss: 0.3911, Val Acc: 86.70%, LR: 0.005450
验证损失从 0.3927 改善到 0.3911，保存模型状态
Epoch 53/60: 100%|███| 352/352 [00:26<00:00, 13.53it/s, loss=0.2793, acc=90.30%]
Validating Epoch 53: 100%|█| 40/40 [00:01<00:00, 31.63it/s, loss=0.4282, acc=85.
Epoch 53/60, Loss: 0.2793, Acc: 90.30%, Val Loss: 0.4282, Val Acc: 85.56%, LR: 0.004323
EarlyStopping 计数器: 1/20
Epoch 54/60: 100%|███| 352/352 [00:25<00:00, 13.68it/s, loss=0.2700, acc=90.60%]
Validating Epoch 54: 100%|█| 40/40 [00:01<00:00, 31.08it/s, loss=0.4026, acc=86.
Epoch 54/60, Loss: 0.2700, Acc: 90.60%, Val Loss: 0.4026, Val Acc: 86.28%, LR: 0.003321
EarlyStopping 计数器: 2/20
Epoch 55/60: 100%|███| 352/352 [00:25<00:00, 13.59it/s, loss=0.2643, acc=90.76%]
Validating Epoch 55: 100%|█| 40/40 [00:01<00:00, 30.34it/s, loss=0.3942, acc=86.
Epoch 55/60, Loss: 0.2643, Acc: 90.76%, Val Loss: 0.3942, Val Acc: 86.62%, LR: 0.002447
EarlyStopping 计数器: 3/20
Epoch 56/60: 100%|███| 352/352 [00:25<00:00, 13.68it/s, loss=0.2535, acc=91.06%]
Validating Epoch 56: 100%|█| 40/40 [00:01<00:00, 30.78it/s, loss=0.3740, acc=86.
Epoch 56/60, Loss: 0.2535, Acc: 91.06%, Val Loss: 0.3740, Val Acc: 86.98%, LR: 0.001704
验证损失从 0.3911 改善到 0.3740，保存模型状态
Epoch 57/60: 100%|███| 352/352 [00:25<00:00, 13.59it/s, loss=0.2483, acc=91.36%]
Validating Epoch 57: 100%|█| 40/40 [00:01<00:00, 32.22it/s, loss=0.3736, acc=87.
Epoch 57/60, Loss: 0.2483, Acc: 91.36%, Val Loss: 0.3736, Val Acc: 87.22%, LR: 0.001093
验证损失从 0.3740 改善到 0.3736，保存模型状态
Epoch 58/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.2427, acc=91.63%]
Validating Epoch 58: 100%|█| 40/40 [00:01<00:00, 32.91it/s, loss=0.3713, acc=87.
Epoch 58/60, Loss: 0.2427, Acc: 91.63%, Val Loss: 0.3713, Val Acc: 87.22%, LR: 0.000616
验证损失从 0.3736 改善到 0.3713，保存模型状态
Epoch 59/60: 100%|███| 352/352 [00:25<00:00, 13.58it/s, loss=0.2373, acc=91.60%]
Validating Epoch 59: 100%|█| 40/40 [00:01<00:00, 31.12it/s, loss=0.3762, acc=87.
Epoch 59/60, Loss: 0.2373, Acc: 91.60%, Val Loss: 0.3762, Val Acc: 87.08%, LR: 0.000274
EarlyStopping 计数器: 1/20
Epoch 60/60: 100%|███| 352/352 [00:25<00:00, 13.69it/s, loss=0.2393, acc=91.66%]
Validating Epoch 60: 100%|█| 40/40 [00:01<00:00, 30.57it/s, loss=0.3812, acc=87.
Epoch 60/60, Loss: 0.2393, Acc: 91.66%, Val Loss: 0.3812, Val Acc: 87.26%, LR: 0.000069
EarlyStopping 计数器: 2/20
训练时间: 1624.33 秒
Evaluating: 100%|██████| 79/79 [00:01<00:00, 39.86it/s, loss=0.3914, acc=87.26%]
Test Loss: 0.3914, Test Accuracy: 87.26%
Training curves saved to results/images/oh_my_resnet_18_training_curves.png
Model saved to results/models/oh_my_resnet_18.pth

结果摘要:
模型: ResNet18
优化器: sgd, 学习率: 0.1, 权重衰减: 0.0005
参数量: 11.17M
训练时间: 1624.33s
最佳验证准确率: 87.26%
测试准确率: 87.26%
测试损失: 0.3914
模型保存为: oh_my_resnet_18.pth

