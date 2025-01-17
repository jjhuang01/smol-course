# PyTorch深度学习基础

## 前言：深度学习之旅
想象你正在学习烹饪：
- PyTorch就像是一个现代化的厨房
- 张量（Tensor）是你的食材
- 模型（Model）是你的菜谱
- 训练（Training）就是不断尝试和改进的过程

## 一、PyTorch基础概念
### 1. 张量（Tensor）：数据的容器
就像超市里的不同包装方式：
- 标量(0维张量)：单个商品，如1个苹果
- 向量(1维张量)：一排商品，如货架上的苹果
- 矩阵(2维张量)：一面货架，多排商品
- 多维张量：整个超市的商品排列

```python
import torch

# 1. 创建张量的多种方式
# 就像采购商品的不同方式
x = torch.tensor([1, 2, 3])  # 直接指定，像是点单
y = torch.zeros(3, 3)        # 全0张量，像是空货架
z = torch.rand(2, 3)         # 随机张量，像是随机采样

# 2. 张量的基本属性
print(f"形状(shape): {x.shape}")      # 像是查看包装规格
print(f"数据类型(dtype): {x.dtype}")  # 像是查看商品类型
print(f"存储设备: {x.device}")        # 像是查看存储位置

# 3. 张量操作
# 就像对商品进行的各种处理
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 数学运算
print(f"加法: {a + b}")              # 像是合并商品
print(f"乘法: {a * b}")              # 像是打包商品
print(f"平均值: {a.mean()}")         # 像是计算均价

# 4. 形状变换
c = torch.tensor([[1, 2, 3], 
                 [4, 5, 6]])
print(f"原始形状: {c.shape}")
c_reshaped = c.reshape(3, 2)         # 像是重新排列货架
print(f"重排后: {c_reshaped.shape}")
c_squeezed = c.squeeze()             # 像是压缩包装
c_unsqueezed = c.unsqueeze(0)        # 像是增加包装层

# 5. GPU加速
if torch.cuda.is_available():
    # 就像把商品从仓库（CPU）转移到店面（GPU）
    x_gpu = x.to('cuda')
    print(f"已转移到: {x_gpu.device}")
```

### 2. 自动求导（Autograd）：梯度的自动计算
就像烹饪时调整配料：
- 前向传播：像是按照配方烹饪
- 反向传播：像是品尝后调整配料比例
- 梯度：像是每种配料需要调整的方向和幅度

```python
# 1. 基础自动求导示例
x = torch.tensor([2.0], requires_grad=True)  # 标记需要求导
y = x * 2 + 1                               # 前向计算
y.backward()                                # 反向传播
print(f"x的梯度: {x.grad}")                 # 查看梯度

# 2. 复杂计算中的自动求导
def complex_function(x):
    """复杂的计算函数示例"""
    return x.sin() * x.exp() + x.pow(3)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = complex_function(x)
y.sum().backward()
print(f"复杂函数的梯度: {x.grad}")

# 3. 梯度累积与清零
x = torch.tensor([1.0], requires_grad=True)
for t in range(3):
    y = x * 2
    y.backward()
    print(f"第{t+1}次累积后的梯度: {x.grad}")
    x.grad.zero_()  # 清零梯度，准备下一次计算

# 4. 防止梯度计算的上下文管理器
with torch.no_grad():
    # 这里的计算不会记录梯度
    y = x * 2
print(f"是否需要梯度: {y.requires_grad}")
```

## 二、神经网络基础
### 1. 构建模型：像搭建积木
就像建造一座房子：
- 线性层（Linear Layer）：像是房子的砖块
- 激活函数（Activation）：像是房间的门，控制信息流动
- 网络结构：像是房子的设计图纸

```python
import torch.nn as nn
import torch.nn.functional as F

# 1. 简单神经网络
class SimpleNN(nn.Module):
    """简单的神经网络，像是一个小房子"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义网络层，像是规划房间
        self.layer1 = nn.Linear(input_size, hidden_size)  # 第一层，像是客厅
        self.layer2 = nn.Linear(hidden_size, output_size) # 第二层，像是卧室
    
    def forward(self, x):
        # 定义数据流动，像是规划参观路线
        x = F.relu(self.layer1(x))  # 第一层激活，像是打开客厅的灯
        x = self.layer2(x)          # 第二层，像是进入卧室
        return x

# 2. 常用层的示例
class CommonLayers(nn.Module):
    """展示常用的网络层，像是不同类型的房间"""
    def __init__(self):
        super().__init__()
        # 全连接层，像是普通房间
        self.fc = nn.Linear(100, 50)
        
        # 卷积层，像是带窗户的房间，可以看到局部特征
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        
        # 池化层，像是压缩空间的魔法房间
        self.pool = nn.MaxPool2d(2)
        
        # Dropout，像是随机关闭的房间，防止过度依赖
        self.dropout = nn.Dropout(0.5)
        
        # 批归一化，像是温度调节器，保持环境稳定
        self.bn = nn.BatchNorm1d(50)
    
    def forward(self, x):
        # 数据依次经过各个房间
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# 3. 激活函数展示
def show_activations(x):
    """展示不同的激活函数，像是不同类型的门"""
    print("ReLU:", F.relu(x))       # 像是单向门，只让正值通过
    print("Sigmoid:", torch.sigmoid(x))  # 像是弹簧门，有缓冲效果
    print("Tanh:", torch.tanh(x))    # 像是旋转门，有正负对称性
```

### 2. 损失函数与优化器：调整与改进
就像是菜品的品尝和改进：
- 损失函数：像是顾客的评分标准
- 优化器：像是主厨根据评分调整配方
- 学习率：像是调整配方的幅度

```python
# 1. 常用损失函数
def loss_functions_demo(pred, target):
    """展示常用的损失函数"""
    # MSE损失，像是测量菜品口感与预期的差距
    mse_loss = nn.MSELoss()(pred, target)
    
    # 交叉熵损失，像是评判菜品类别是否正确
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    
    # L1损失，像是测量配料用量的偏差
    l1_loss = nn.L1Loss()(pred, target)
    
    return mse_loss, ce_loss, l1_loss

# 2. 优化器示例
def optimizers_demo(model):
    """展示不同的优化器"""
    # SGD优化器，像是最基础的调整方法
    sgd = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Adam优化器，像是智能的主厨，能自适应调整
    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # RMSprop优化器，像是有经验的主厨，重视近期反馈
    rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
    return sgd, adam, rmsprop

# 3. 学习率调整
def lr_scheduler_demo(optimizer):
    """展示学习率调整策略"""
    # 步进式调整，像是定期调整配方
    step_lr = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)
    
    # 动态调整，像是根据效果动态调整配方
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10)
    
    return step_lr, reduce_lr
```

## 三、模型训练实战
### 1. 训练流程：完整的学习过程
就像是开餐厅的完整流程：
- 数据准备：像是采购食材
- 训练循环：像是不断练习烹饪
- 验证评估：像是试菜环节

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """完整的模型训练流程"""
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # 记录训练历史
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss,
            'accuracy': accuracy
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return training_history

# 使用示例
def training_example():
    # 1. 准备数据
    train_loader = get_train_data()  # 获取训练数据
    val_loader = get_val_data()      # 获取验证数据
    
    # 2. 创建模型
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, epochs=10
    )
    
    # 5. 可视化训练过程
    plot_training_history(history)
```

### 2. 模型评估与调试
就像是餐厅的质量控制：
- 性能指标：像是顾客评价
- 过拟合处理：像是避免菜品过于特殊化
- 调试技巧：像是解决烹饪问题

```python
def model_evaluation_tools():
    """模型评估和调试的工具集"""
    
    def evaluate_metrics(model, test_loader):
        """计算各种评估指标"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # 计算各种指标
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted')
        recall = recall_score(targets, predictions, average='weighted')
        f1 = f1_score(targets, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def debug_gradients(model):
        """检查梯度问题"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"梯度范数: {param.grad.norm().item()}")
                print(f"参数范数: {param.data.norm().item()}")
                print("---")
    
    def check_overfitting(train_history, val_history):
        """检查过拟合"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label='训练损失')
        plt.plot(val_history, label='验证损失')
        plt.title('损失曲线')
        plt.legend()
        plt.show()
```

## 四、实战项目：手写数字识别
让我们把学到的知识应用到实际项目中：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DigitRecognizer(nn.Module):
    """手写数字识别模型"""
    def __init__(self):
        super().__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(1, 32, 3)  # 第一层卷积
        self.conv2 = nn.Conv2d(32, 64, 3)  # 第二层卷积
        self.pool = nn.MaxPool2d(2)  # 池化层
        self.dropout1 = nn.Dropout2d(0.25)  # 防止过拟合
        
        # 分类层
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全连接层
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # 输出层
    
    def forward(self, x):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 分类
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def train_digit_recognizer():
    """训练手写数字识别模型的完整流程"""
    # 1. 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True,
                                 transform=transform)
    test_dataset = datasets.MNIST('data', train=False,
                                transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 2. 创建模型
    model = DigitRecognizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 3. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练循环
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
        
        # 5. 评估
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%)')

if __name__ == '__main__':
    train_digit_recognizer()
```

## 五、进阶主题
### 1. 迁移学习
就像是在已有餐厅的基础上开新店：
- 预训练模型：像是成熟的菜谱
- 微调：像是根据本地口味调整

### 2. 模型部署
就像是把实验厨房的菜品推广到连锁店：
- 模型导出：像是标准化菜谱
- 性能优化：像是提高出菜效率
- 部署环境：像是适应不同的厨房条件

### 3. 性能优化
就像是提高餐厅效率：
- 批处理：像是批量烹饪
- 并行计算：像是多个厨师同时工作
- 内存管理：像是合理利用厨房空间

## 六、常见问题与解决方案
1. 梯度消失/爆炸
   - 问题：像是调料比例失控
   - 解决：使用批归一化、残差连接等

2. 过拟合
   - 问题：像是菜品过于迎合某些顾客
   - 解决：数据增强、Dropout、正则化

3. 训练不稳定
   - 问题：像是烹饪效果忽好忽坏
   - 解决：调整学习率、使用更好的优化器

## 七、实践建议
1. 从小数据集开始练习
2. 多观察损失曲线变化
3. 保持代码的模块化和可读性
4. 建立实验记录习惯
5. 循序渐进，由简单到复杂 