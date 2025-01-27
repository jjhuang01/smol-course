# AI模型部署与工程实践

## 一、模型部署基础

### 1. 什么是模型部署？
想象你是一个厨师：
- 研发新菜品就像训练模型
- 把菜品端到客人面前就像部署模型
- 需要考虑口感、温度、摆盘等细节

模型部署就是：
- 把训练好的模型投入使用
- 让用户能够方便地访问
- 确保性能和稳定性

### 2. 部署架构
```python
class ModelDeployment:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = torch.load(model_path)
        self.model.eval()
    
    def preprocess(self, data):
        """数据预处理"""
        return self.preprocessor.transform(data)
    
    def postprocess(self, predictions):
        """结果后处理"""
        return self.postprocessor.transform(predictions)
    
    def predict(self, data):
        """预测流程"""
        processed_data = self.preprocess(data)
        with torch.no_grad():
            predictions = self.model(processed_data)
        return self.postprocess(predictions)
```

## 二、部署方式

### 1. RESTful API
使用FastAPI构建API服务：
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list
    parameters: dict = {}

class ModelService:
    def __init__(self):
        self.model = load_model()
    
    async def predict(self, data):
        return self.model.predict(data)

model_service = ModelService()

@app.post("/predict")
async def predict(request: PredictionRequest):
    predictions = await model_service.predict(request.data)
    return {"predictions": predictions}
```

### 2. gRPC服务
使用gRPC进行高性能通信：
```python
import grpc
from concurrent import futures
import prediction_pb2
import prediction_pb2_grpc

class PredictionService(prediction_pb2_grpc.PredictionServicer):
    def __init__(self):
        self.model = load_model()
    
    def Predict(self, request, context):
        predictions = self.model.predict(request.data)
        return prediction_pb2.PredictionResponse(
            predictions=predictions
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictionServicer_to_server(
        PredictionService(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

### 3. 批处理服务
处理大批量预测请求：
```python
class BatchProcessor:
    def __init__(self, batch_size=32):
        self.model = load_model()
        self.batch_size = batch_size
        self.queue = Queue()
    
    def add_task(self, data):
        self.queue.put(data)
    
    def process_batch(self):
        batch = []
        while len(batch) < self.batch_size and not self.queue.empty():
            batch.append(self.queue.get())
        
        if batch:
            return self.model.predict(batch)
        return None
    
    def run(self):
        while True:
            predictions = self.process_batch()
            if predictions is not None:
                self.save_results(predictions)
            time.sleep(0.1)
```

## 三、性能优化

### 1. 模型优化
```python
class ModelOptimizer:
    def __init__(self, model):
        self.model = model
    
    def quantize(self):
        """模型量化"""
        return torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    def prune(self, amount=0.3):
        """模型剪枝"""
        parameters_to_prune = [
            (module, 'weight')
            for module in self.model.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=amount
        )
    
    def optimize_for_inference(self):
        """优化推理"""
        self.model = torch.jit.script(self.model)
        return self.model
```

### 2. 并发处理
```python
class ConcurrentPredictor:
    def __init__(self, model, num_workers=4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    async def predict_batch(self, batch):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.model.predict,
            batch
        )
    
    async def predict_many(self, data_list):
        tasks = [
            self.predict_batch(batch)
            for batch in self.create_batches(data_list)
        ]
        return await asyncio.gather(*tasks)
```

## 四、监控与维护

### 1. 性能监控
```python
class ModelMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_prediction(self, latency, memory_usage):
        self.metrics['latency'].append(latency)
        self.metrics['memory'].append(memory_usage)
    
    def log_error(self, error):
        self.metrics['errors'].append({
            'time': time.time(),
            'error': str(error)
        })
    
    def get_statistics(self):
        return {
            'avg_latency': np.mean(self.metrics['latency']),
            'p95_latency': np.percentile(self.metrics['latency'], 95),
            'error_rate': len(self.metrics['errors']) / len(self.metrics['latency'])
        }
```

### 2. 日志管理
```python
class LogManager:
    def __init__(self, log_path):
        self.log_path = log_path
        self.logger = self.setup_logger()
    
    def setup_logger(self):
        logger = logging.getLogger('model_service')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(handler)
        return logger
    
    def log_prediction(self, request_id, input_data, output, latency):
        self.logger.info(
            f"Request {request_id}: "
            f"Input shape {input_data.shape}, "
            f"Output shape {output.shape}, "
            f"Latency {latency:.2f}ms"
        )
```

## 五、扩展与负载均衡

### 1. 水平扩展
```python
class ModelCluster:
    def __init__(self, num_replicas=3):
        self.replicas = [
            ModelService()
            for _ in range(num_replicas)
        ]
        self.current_replica = 0
    
    def get_next_replica(self):
        """轮询调度"""
        replica = self.replicas[self.current_replica]
        self.current_replica = (self.current_replica + 1) % len(self.replicas)
        return replica
    
    async def predict(self, data):
        replica = self.get_next_replica()
        return await replica.predict(data)
```

### 2. 负载均衡
```python
class LoadBalancer:
    def __init__(self, services):
        self.services = services
        self.weights = [1.0] * len(services)
    
    def update_weights(self, latencies):
        """基于延迟更新权重"""
        total_latency = sum(latencies)
        for i, latency in enumerate(latencies):
            self.weights[i] = 1.0 / (latency / total_latency)
    
    def select_service(self):
        """加权随机选择"""
        return random.choices(
            self.services,
            weights=self.weights,
            k=1
        )[0]
```

## 六、容错与恢复

### 1. 熔断机制
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_time=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.last_failure_time = None
        self.is_open = False
    
    async def call(self, func, *args, **kwargs):
        if self.is_open:
            if time.time() - self.last_failure_time > self.recovery_time:
                self.is_open = False
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                self.last_failure_time = time.time()
            raise e
```

### 2. 备份恢复
```python
class ModelBackup:
    def __init__(self, model, backup_dir):
        self.model = model
        self.backup_dir = backup_dir
    
    def save_checkpoint(self):
        """保存模型检查点"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'timestamp': time.time()
        }
        path = os.path.join(
            self.backup_dir,
            f"checkpoint_{int(time.time())}.pt"
        )
        torch.save(checkpoint, path)
    
    def restore_from_checkpoint(self, checkpoint_path):
        """从检查点恢复"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
```

## 七、版本管理与更新

### 1. 模型版本控制
```python
class ModelVersionControl:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.versions = {}
    
    def save_version(self, model, version, metadata=None):
        """保存模型版本"""
        version_path = os.path.join(
            self.storage_path,
            f"model_v{version}"
        )
        os.makedirs(version_path, exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), 
                  os.path.join(version_path, "model.pt"))
        
        # 保存元数据
        if metadata:
            with open(os.path.join(version_path, "metadata.json"), "w") as f:
                json.dump(metadata, f)
        
        self.versions[version] = {
            'path': version_path,
            'metadata': metadata,
            'timestamp': time.time()
        }
    
    def load_version(self, version):
        """加载特定版本"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        version_info = self.versions[version]
        model = torch.load(
            os.path.join(version_info['path'], "model.pt")
        )
        return model, version_info['metadata']
```

### 2. 在线更新
```python
class OnlineUpdater:
    def __init__(self, model_service):
        self.model_service = model_service
        self.update_lock = asyncio.Lock()
    
    async def update_model(self, new_model):
        """在线更新模型"""
        async with self.update_lock:
            # 保存旧模型作为备份
            old_model = self.model_service.model
            
            try:
                # 更新模型
                self.model_service.model = new_model
                
                # 验证新模型
                validation_result = await self.validate_model(new_model)
                if not validation_result:
                    # 如果验证失败，回滚到旧模型
                    self.model_service.model = old_model
                    raise ValueError("Model validation failed")
                
            except Exception as e:
                # 发生错误时回滚
                self.model_service.model = old_model
                raise e
    
    async def validate_model(self, model):
        """验证模型性能"""
        test_data = self.load_test_data()
        predictions = await self.model_service.predict(test_data)
        return self.evaluate_predictions(predictions)
```

## 八、安全性考虑

### 1. 输入验证
```python
class InputValidator:
    def __init__(self):
        self.validators = {
            'image': self.validate_image,
            'text': self.validate_text,
            'numeric': self.validate_numeric
        }
    
    def validate_image(self, image):
        """验证图像输入"""
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid image format")
        
        if image.shape[-1] not in [1, 3, 4]:
            raise ValueError("Invalid number of channels")
        
        if image.dtype != np.uint8:
            raise ValueError("Invalid image dtype")
    
    def validate_text(self, text):
        """验证文本输入"""
        if not isinstance(text, str):
            raise ValueError("Input must be string")
        
        if len(text) > 1000:  # 最大长度限制
            raise ValueError("Text too long")
        
        # 检查特殊字符
        if re.search(r'[<>]', text):
            raise ValueError("Invalid characters in text")
    
    def validate_numeric(self, data):
        """验证数值输入"""
        if not isinstance(data, (np.ndarray, list)):
            raise ValueError("Invalid numeric format")
        
        data = np.array(data)
        if not np.isfinite(data).all():
            raise ValueError("Invalid numeric values")
```

### 2. 访问控制
```python
class SecurityManager:
    def __init__(self):
        self.api_keys = {}
        self.rate_limiters = {}
    
    def authenticate(self, api_key):
        """验证API密钥"""
        if api_key not in self.api_keys:
            raise ValueError("Invalid API key")
        
        if self.is_rate_limited(api_key):
            raise ValueError("Rate limit exceeded")
    
    def is_rate_limited(self, api_key):
        """检查速率限制"""
        if api_key not in self.rate_limiters:
            self.rate_limiters[api_key] = {
                'count': 0,
                'last_reset': time.time()
            }
        
        limiter = self.rate_limiters[api_key]
        
        # 重置计数器
        if time.time() - limiter['last_reset'] > 3600:
            limiter['count'] = 0
            limiter['last_reset'] = time.time()
        
        # 检查限制
        if limiter['count'] >= 1000:  # 每小时1000次请求
            return True
        
        limiter['count'] += 1
        return False
```

## 九、最佳实践

### 1. 部署清单
```python
class DeploymentChecklist:
    def __init__(self):
        self.checks = {
            'model': self.check_model,
            'performance': self.check_performance,
            'security': self.check_security,
            'monitoring': self.check_monitoring
        }
    
    def check_model(self):
        """检查模型准备情况"""
        checks = [
            "模型已经过充分测试",
            "模型大小适合部署环境",
            "模型格式正确",
            "依赖项已解决"
        ]
        return self.run_checks(checks)
    
    def check_performance(self):
        """检查性能要求"""
        checks = [
            "延迟满足要求",
            "吞吐量满足要求",
            "资源使用合理",
            "扩展性方案就绪"
        ]
        return self.run_checks(checks)
    
    def check_security(self):
        """检查安全措施"""
        checks = [
            "输入验证完善",
            "访问控制就绪",
            "数据加密保护",
            "审计日志完备"
        ]
        return self.run_checks(checks)
    
    def check_monitoring(self):
        """检查监控系统"""
        checks = [
            "性能监控就绪",
            "错误追踪系统",
            "告警机制完善",
            "日志系统配置"
        ]
        return self.run_checks(checks)
```

### 2. 部署流程
```python
class DeploymentPipeline:
    def __init__(self):
        self.stages = [
            self.prepare_model,
            self.setup_infrastructure,
            self.deploy_service,
            self.verify_deployment
        ]
    
    async def run(self):
        """执行部署流程"""
        for stage in self.stages:
            try:
                await stage()
            except Exception as e:
                await self.rollback()
                raise e
    
    async def prepare_model(self):
        """准备模型"""
        # 优化模型
        optimizer = ModelOptimizer(self.model)
        self.optimized_model = optimizer.optimize_for_inference()
        
        # 验证模型
        validator = ModelValidator()
        await validator.validate(self.optimized_model)
    
    async def setup_infrastructure(self):
        """设置基础设施"""
        # 配置服务器
        self.server = ModelServer()
        await self.server.initialize()
        
        # 设置监控
        self.monitor = ModelMonitor()
        await self.monitor.start()
    
    async def deploy_service(self):
        """部署服务"""
        # 部署模型
        self.service = ModelService(self.optimized_model)
        await self.service.start()
        
        # 配置负载均衡
        self.load_balancer = LoadBalancer([self.service])
        await self.load_balancer.start()
    
    async def verify_deployment(self):
        """验证部署"""
        # 运行测试
        test_runner = TestRunner()
        await test_runner.run_tests()
        
        # 检查性能
        performance = await self.monitor.check_performance()
        if not performance.is_satisfactory():
            raise ValueError("Performance verification failed")
```

### 3. 维护指南
```python
class MaintenanceGuide:
    def __init__(self):
        self.tasks = {
            'daily': self.daily_tasks,
            'weekly': self.weekly_tasks,
            'monthly': self.monthly_tasks
        }
    
    async def daily_tasks(self):
        """日常维护任务"""
        # 检查性能指标
        await self.check_performance_metrics()
        
        # 检查错误日志
        await self.analyze_error_logs()
        
        # 备份关键数据
        await self.backup_critical_data()
    
    async def weekly_tasks(self):
        """每周维护任务"""
        # 性能优化
        await self.optimize_performance()
        
        # 更新模型（如果需要）
        await self.update_model_if_needed()
        
        # 清理旧日志和数据
        await self.cleanup_old_data()
    
    async def monthly_tasks(self):
        """每月维护任务"""
        # 全面系统检查
        await self.full_system_check()
        
        # 安全审计
        await self.security_audit()
        
        # 容量规划
        await self.capacity_planning()