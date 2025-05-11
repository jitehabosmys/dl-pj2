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


