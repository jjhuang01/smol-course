# AI实战项目指南

## 一、项目规划与准备
### 1. 项目选择
- 🎯 难度递进：从简单到复杂
- 🌟 领域选择：CV、NLP、RL等
- 📊 评估标准：
  - 技术可行性
  - 数据可获得性
  - 计算资源需求

### 2. 环境配置
```bash
# 创建虚拟环境
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# 安装基础包
pip install torch torchvision
pip install transformers datasets
pip install pandas numpy matplotlib
```

## 二、实战项目示例
### 1. 图像分类器
```python
import torch
import torchvision
from torch import nn
from torchvision import transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 加载数据
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                        shuffle=True, num_workers=2)

# 模型定义
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # 修改最后一层

# 训练循环
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 2. 文本分类器
```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# 数据准备
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 模型训练
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 图像生成器
```python
# 简单的GAN实现
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

# 训练过程
def train_gan(generator, discriminator, dataloader):
    g_optimizer = torch.optim.Adam(generator.parameters())
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    
    for epoch in range(100):
        for real_images in dataloader:
            # 训练判别器
            d_optimizer.zero_grad()
            batch_size = real_images.size(0)
            label_real = torch.ones(batch_size, 1)
            label_fake = torch.zeros(batch_size, 1)
            
            d_loss_real = criterion(discriminator(real_images), label_real)
            
            noise = torch.randn(batch_size, 100)
            fake_images = generator(noise)
            d_loss_fake = criterion(discriminator(fake_images.detach()), label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images), label_real)
            g_loss.backward()
            g_optimizer.step()
```

## 三、项目优化技巧
### 1. 性能优化
- 🔧 数据加载优化
  - 使用 `num_workers`
  - 适当的 `batch_size`
  - 数据预取

- 🚀 模型加速
```python
# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 模型调优
- 📈 学习率调整
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

- 🎯 正则化技术
```python
# 添加权重衰减
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

# Dropout层
self.dropout = nn.Dropout(0.5)
```

## 四、部署与服务化
### 1. 模型导出
```python
# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pth')

# 加载模型
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. Web服务部署
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # 预处理
    input_tensor = preprocess(data)
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    # 后处理
    result = postprocess(output)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 五、项目管理最佳实践
1. 代码组织
   ```
   project/
   ├── data/
   ├── models/
   ├── config/
   ├── utils/
   ├── train.py
   ├── evaluate.py
   └── README.md
   ```

2. 实验记录
   - 使用 MLflow 或 Weights & Biases
   - 记录超参数和结果
   - 保存模型检查点

3. 测试与评估
   ```python
   # 单元测试
   def test_model_output():
       model = YourModel()
       x = torch.randn(1, 3, 224, 224)
       output = model(x)
       assert output.shape == (1, num_classes)
   ```

## 六、常见问题解决
1. 内存管理
   - 使用 `del` 释放不需要的变量
   - 定期清理GPU缓存
   - 使用梯度累积处理大批量

2. 调试技巧
   ```python
   # 打印中间层输出
   class DebugLayer(nn.Module):
       def forward(self, x):
           print(f"Shape: {x.shape}, Mean: {x.mean()}")
           return x
   ```

## 学习路径建议
1. 基础项目
   - MNIST手写数字识别
   - 情感分析
   - 简单图像分类

2. 进阶项目
   - 目标检测
   - 机器翻译
   - 图像生成

3. 高级项目
   - 多模态学习
   - 强化学习
   - 自动驾驶 