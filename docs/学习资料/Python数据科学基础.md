# Python数据科学基础

## 一、Python基础语法
### 1. 数据类型与变量
就像在厨房准备食材：
- 数字(int/float)：像是配料的重量（克、毫升）
- 字符串(str)：像是菜谱的步骤说明
- 列表(list)：像是一个可以随时添加或取出食材的购物篮
- 字典(dict)：像是一个标注了位置的调料架
- 元组(tuple)：像是一个密封好的、不能更改的调味料包

```python
# 1. 数值运算
weight = 75.5  # 体重（千克）
height = 1.75  # 身高（米）
bmi = weight / (height ** 2)
print(f"BMI指数: {bmi:.2f}")

# 2. 字符串处理
name = "张三"
age = 25
intro = f"{name}今年{age}岁"  # f-string格式化
print(intro.replace("张三", "李四"))  # 字符串替换

# 3. 列表操作
scores = [85, 92, 78, 95, 88]
scores.append(90)  # 添加新成绩
average = sum(scores) / len(scores)
print(f"平均分: {average:.1f}")

# 4. 字典使用
student = {
    "name": "张三",
    "scores": {
        "数学": 85,
        "英语": 92,
        "Python": 95
    }
}
print(f"Python成绩: {student['scores']['Python']}")
```

### 2. 控制流程
就像做菜的步骤和判断：
- if-else：像是判断食材是否新鲜
- for循环：像是批量处理多个菜品
- while循环：像是炒菜时不断翻炒直到熟透
- try-except：像是处理意外情况（如锅烧干了）

```python
# 1. 条件判断
def check_score(score):
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 60:
        return "及格"
    else:
        return "需要努力"

# 2. for循环示例
def analyze_scores(scores):
    total = 0
    highest = float('-inf')
    for score in scores:
        total += score
        highest = max(highest, score)
    return {
        "平均分": total / len(scores),
        "最高分": highest
    }

# 3. while循环示例
def find_target(numbers, target):
    left, right = 0, len(numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if numbers[mid] == target:
            return mid
        elif numbers[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 4. 异常处理
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "除数不能为0"
    except TypeError:
        return "输入必须是数字"
```

### 3. 函数与模块
就像厨师的不同技能和工具：
- 函数：像是特定的烹饪技巧
- 模块：像是不同的厨具套装
- 类：像是一个完整的菜品制作流程

```python
# 1. 函数定义与使用
def calculate_bmi(weight: float, height: float) -> float:
    """
    计算BMI指数
    
    参数:
        weight: 体重(kg)
        height: 身高(m)
    返回:
        bmi: BMI指数
    """
    return weight / (height ** 2)

# 2. 装饰器示例
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"函数返回: {result}")
        return result
    return wrapper

@log_function_call
def add_numbers(a, b):
    return a + b

# 3. 类的定义与使用
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.scores = []
    
    def add_score(self, score):
        self.scores.append(score)
    
    def get_average(self):
        return sum(self.scores) / len(self.scores)
    
    @property
    def status(self):
        avg = self.get_average()
        return "优秀" if avg >= 90 else "良好" if avg >= 80 else "及格"
```

## 二、NumPy数组操作
### 1. 数组基础
就像在超市整理商品：
- 创建数组：像是进货上架
- 索引切片：像是找到特定位置的商品
- 形状变换：像是重新排列商品

```python
import numpy as np

# 1. 创建数组
arr1 = np.array([1, 2, 3, 4, 5])  # 一维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 二维数组
zeros = np.zeros((3, 3))  # 全0数组
ones = np.ones((2, 4))  # 全1数组
rand_arr = np.random.rand(3, 3)  # 随机数组

# 2. 索引与切片
print(arr2[0, 1])  # 访问单个元素
print(arr2[:, 1])  # 访问整列
print(arr2[0, :])  # 访问整行

# 3. 形状操作
arr3 = arr2.reshape(3, 2)  # 重塑形状
arr4 = arr2.T  # 转置
```

### 2. 数组运算
就像批量处理商品：
- 数组计算：像是批量定价
- 统计函数：像是销售统计
- 广播机制：像是不同规格商品的价格计算

```python
# 1. 基础运算
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)  # 加法
print(arr1 * 2)  # 标量乘法
print(arr1 * arr2)  # 元素乘法
print(np.dot(arr1, arr2))  # 点积

# 2. 统计运算
data = np.random.randn(1000)  # 生成随机数据
print(f"平均值: {data.mean():.2f}")
print(f"标准差: {data.std():.2f}")
print(f"最大值: {data.max():.2f}")
print(f"最小值: {data.min():.2f}")

# 3. 广播示例
heights = np.random.normal(170, 10, 100)  # 身高数据
weights = np.random.normal(65, 15, 100)  # 体重数据
bmi = weights / (heights/100) ** 2  # 广播计算BMI
```

## 三、Pandas数据处理
### 1. Series与DataFrame
就像管理学生成绩单：
- Series：像是单科成绩列表
- DataFrame：像是完整的成绩单
- 索引：像是学号或姓名查找

```python
import pandas as pd

# 1. Series示例
grades = pd.Series([85, 92, 78, 95, 88], 
                  index=['张三', '李四', '王五', '赵六', '钱七'],
                  name='数学成绩')
print(grades['张三'])  # 通过索引访问
print(grades[grades >= 90])  # 条件筛选

# 2. DataFrame示例
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [20, 21, 19, 22],
    '数学': [85, 92, 78, 95],
    '英语': [88, 95, 82, 85],
    'Python': [92, 88, 85, 90]
}
df = pd.DataFrame(data)
print(df.describe())  # 统计描述
print(df.sort_values('数学', ascending=False))  # 排序
```

### 2. 数据处理
就像整理和分析成绩单：
- 清洗：处理缺失和异常值
- 转换：计算平均分、排名
- 聚合：按班级统计成绩

```python
# 1. 数据清洗
def clean_student_data(df):
    # 处理缺失值
    df['成绩'].fillna(df['成绩'].mean(), inplace=True)
    
    # 处理异常值
    df.loc[df['成绩'] > 100, '成绩'] = 100
    df.loc[df['成绩'] < 0, '成绩'] = 0
    
    return df

# 2. 数据转换
def transform_data(df):
    # 添加新列
    df['平均分'] = df[['数学', '英语', 'Python']].mean(axis=1)
    df['是否及格'] = df['平均分'].apply(lambda x: '是' if x >= 60 else '否')
    
    # 数据标准化
    df['数学_标准分'] = (df['数学'] - df['数学'].mean()) / df['数学'].std()
    
    return df

# 3. 数据聚合
def aggregate_data(df):
    # 按班级统计
    class_stats = df.groupby('班级').agg({
        '数学': ['mean', 'std', 'min', 'max'],
        '英语': ['mean', 'std', 'min', 'max'],
        'Python': ['mean', 'std', 'min', 'max']
    })
    
    return class_stats
```

## 四、数据可视化
### 1. Matplotlib基础
就像把数据画成图表：
- 折线图：展示趋势
- 柱状图：对比数值
- 散点图：查看关系
- 饼图：显示占比

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 基础图表
def plot_basic_charts(data):
    plt.figure(figsize=(15, 10))
    
    # 折线图
    plt.subplot(2, 2, 1)
    plt.plot(data['时间'], data['温度'], marker='o')
    plt.title('温度变化趋势')
    
    # 柱状图
    plt.subplot(2, 2, 2)
    plt.bar(data['科目'], data['成绩'])
    plt.title('各科成绩对比')
    
    # 散点图
    plt.subplot(2, 2, 3)
    plt.scatter(data['身高'], data['体重'])
    plt.title('身高体重关系')
    
    # 饼图
    plt.subplot(2, 2, 4)
    plt.pie(data['比例'], labels=data['类别'], autopct='%1.1f%%')
    plt.title('各类别占比')
    
    plt.tight_layout()
    plt.show()

# 2. 高级可视化
def plot_advanced_charts(df):
    # 设置风格
    plt.style.use('seaborn')
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 箱线图
    sns.boxplot(data=df, ax=axes[0, 0])
    axes[0, 0].set_title('成绩分布')
    
    # 小提琴图
    sns.violinplot(data=df, ax=axes[0, 1])
    axes[0, 1].set_title('成绩密度分布')
    
    # 热力图
    sns.heatmap(df.corr(), annot=True, ax=axes[1, 0])
    axes[1, 0].set_title('相关性热力图')
    
    # KDE图
    sns.kdeplot(data=df, ax=axes[1, 1])
    axes[1, 1].set_title('核密度估计')
    
    plt.tight_layout()
    plt.show()
```

## 五、实战项目示例
### 1. 学生成绩分析系统
```python
class GradeAnalysisSystem:
    def __init__(self):
        self.df = None
    
    def load_data(self, file_path):
        """加载数据"""
        self.df = pd.read_csv(file_path)
    
    def clean_data(self):
        """数据清洗"""
        self.df = clean_student_data(self.df)
    
    def analyze_data(self):
        """数据分析"""
        stats = {
            '总体统计': self.df.describe(),
            '班级统计': aggregate_data(self.df),
            '及格率': (self.df['平均分'] >= 60).mean()
        }
        return stats
    
    def visualize_data(self):
        """数据可视化"""
        plot_advanced_charts(self.df)
    
    def generate_report(self):
        """生成报告"""
        stats = self.analyze_data()
        print("=== 成绩分析报告 ===")
        print("\n1. 总体情况:")
        print(stats['总体统计'])
        print("\n2. 班级情况:")
        print(stats['班级统计'])
        print(f"\n3. 及格率: {stats['及格率']:.2%}")
        self.visualize_data()

# 使用示例
if __name__ == "__main__":
    system = GradeAnalysisSystem()
    system.load_data('grades.csv')
    system.clean_data()
    system.generate_report()
```

## 六、练习与实践
### 1. 基础练习
1. 实现一个函数计算给定数列的统计指标（均值、中位数、标准差等）
2. 处理缺失值和异常值
3. 使用不同类型的图表可视化数据

### 2. 进阶项目
1. 创建一个完整的数据分析流程
2. 实现数据的导入、清洗、分析和可视化
3. 生成分析报告

### 3. 实战挑战
1. 分析真实数据集
2. 解决实际问题
3. 优化代码性能 