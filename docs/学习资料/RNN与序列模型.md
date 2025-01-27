# 连续烹饪艺术：RNN与序列模型 🔄

> 就像烹饪需要按步骤进行，RNN也是处理序列数据的大厨！

## 📚 基础概念

### 1. RNN结构
- 输入门：接收新食材
- 隐藏状态：保存烹饪进度
- 输出门：产出当前结果

### 2. LSTM原理
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        
    def forward(self, x, hidden):
        # x: 输入序列
        # hidden: (h0, c0) 初始状态
        output, (hn, cn) = self.lstm(x, hidden)
        return output, (hn, cn)
```

### 3. GRU特点
- 更新门：调整配料比例
- 重置门：决定是否重新开始
- 计算效率：简化版的LSTM

## 🔧 实现技巧

### 1. 序列处理
- 数据预处理
- 序列填充
- 长度归一化

### 2. 训练策略
- 梯度裁剪
- 教师强制
- 注意力机制

### 3. 优化方法
- 双向RNN
- 多层堆叠
- 残差连接

## 🎯 应用场景

### 1. 文本生成
- 菜谱生成
- 配料描述
- 烹饪步骤

### 2. 时序预测
- 销量预测
- 库存管理
- 价格趋势

### 3. 序列标注
- 配料识别
- 步骤分类
- 关键点标注

## 📈 模型对比

### 1. 结构对比
| 模型 | 参数量 | 计算复杂度 |
|------|--------|------------|
| SimpleRNN | 低 | 低 |
| LSTM | 高 | 中 |
| GRU | 中 | 中 |

### 2. 性能对比
| 模型 | 长期依赖 | 训练速度 |
|------|----------|----------|
| SimpleRNN | 弱 | 快 |
| LSTM | 强 | 慢 |
| GRU | 中 | 中 |

## 💡 实战技巧

### 1. 数据准备
```python
# 序列填充示例
def pad_sequence(sequences, max_len):
    padded = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded
```

### 2. 模型训练
```python
# 梯度裁剪示例
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. 预测优化
```python
# 集成预测示例
def ensemble_predict(models, x):
    predictions = [model(x) for model in models]
    return torch.mean(torch.stack(predictions), dim=0)
```

## 🚀 进阶主题

### 1. 注意力机制
- 自注意力
- 多头注意力
- 位置编码

### 2. Transformer
- 编码器-解码器
- 残差连接
- Layer Normalization

### 3. 混合架构
- CNN+RNN
- RNN+Attention
- Transformer变体 