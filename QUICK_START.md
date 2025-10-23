# 🚀 快速开始：添加新算子测试

## 📋 概述

本项目提供了**完全模块化**的性能测试框架。添加新算子时，你只需要写**最少的代码**：

- ✅ **自动缓存清理**：每次测试前自动清空 NPU/CUDA 缓存
- ✅ **自动性能测试**：自动执行 warmup 和多次测试
- ✅ **自动统计计算**：自动计算 mean, median, std, min, max
- ✅ **自动进度显示**：每 10 次显示一次进度
- ✅ **自动结果保存**：自动生成 JSON 格式结果文件

**你只需要实现 3 个简单方法！**

---

## 🎯 添加新算子的 3 步流程

### Step 1: 复制模板

```bash
# 例如：添加一个名为 "relu" 的新算子
cp TEMPLATE_custom_test.py op/relu/relu_custom_test.py
```

### Step 2: 查找替换

在 `relu_custom_test.py` 中：

```python
# 全局替换（区分大小写）
{OPERATOR_NAME} → relu
{OPERATOR_NAME_CAPITALIZED} → Relu
```

使用编辑器的查找替换功能：
- VSCode: `Ctrl/Cmd + H`
- Vim: `:%s/{OPERATOR_NAME}/relu/g`

### Step 3: 实现 3 个方法

#### 3.1 存储输入（`__init__`）

```python
def __init__(self, x):
    super().__init__()
    self.x = x  # 存储你的输入
```

#### 3.2 运行自定义算子（`run_operator`）

```python
def run_operator(self, *inputs):
    if inputs:
        return relu_custom.run_relu_custom(*inputs)
    return relu_custom.run_relu_custom(self.x)
```

#### 3.3 运行参考实现（`run_reference`）

```python
def run_reference(self, *inputs):
    x_ref = inputs[0] if inputs else self.x.cpu()
    return torch.relu(x_ref)
```

#### 3.4 准备测试输入

```python
def test_relu_custom_ops(self):
    # 准备输入数据
    x = torch.rand([1024, 1024], dtype=torch.float16).npu()

    # 创建测试实例
    perf_test = ReluPerformanceTest(x)
    perf_test.setup_device()

    # 正确性测试
    output = perf_test.run_operator()
    cpuout = torch.relu(x.cpu())
    self.assertRtolEqual(output, cpuout)

    # 性能测试（自动完成！）
    results = perf_test.measure_performance()
    perf_test.save_results(results)
```

### 完成！🎉

运行测试：
```bash
python evaluater.py --op relu --runs 20
```

---

## 📚 完整示例对比

### ❌ 旧方式（手动实现，~100 行）

```python
# 需要手写所有逻辑
for i in range(NUM_TRIALS):
    # 手动清理缓存
    torch_npu.npu.empty_cache()
    torch_npu.npu.synchronize()

    # 手动计时
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)
    start_event.record()
    output = relu_custom.run_relu_custom(x)
    end_event.record()
    torch_npu.npu.synchronize()

    # 手动收集时间
    elapsed_time = start_event.elapsed_time(end_event)
    times.append(elapsed_time)

    # 手动显示进度
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{NUM_TRIALS}")

# 手动计算统计
mean = sum(times) / len(times)
median = sorted(times)[len(times)//2]
std = (sum((x - mean)**2 for x in times) / len(times)) ** 0.5
# ... 更多统计代码

# 手动保存结果
with open('timing_results.json', 'w') as f:
    json.dump({'mean': mean, 'median': median, ...}, f)
```

### ✅ 新方式（使用模块，~30 行）

```python
class ReluPerformanceTest(BasePerformanceTest):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def run_operator(self, *inputs):
        return relu_custom.run_relu_custom(self.x)

    def run_reference(self, *inputs):
        return torch.relu(self.x.cpu())

class TestCustomRelu(TestCase):
    def test_relu_custom_ops(self):
        x = torch.rand([1024, 1024], dtype=torch.float16).npu()
        perf_test = ReluPerformanceTest(x)
        perf_test.setup_device()

        # 正确性
        output = perf_test.run_operator()
        cpuout = torch.relu(x.cpu())
        self.assertRtolEqual(output, cpuout)

        # 性能（自动！）
        results = perf_test.measure_performance()
        perf_test.save_results(results)
```

**代码量减少 70%！所有复杂逻辑都在 `BasePerformanceTest` 中！**

---

## 🔧 高级功能

### 自定义 warmup 次数

```python
perf_test = ReluPerformanceTest(x)
perf_test.num_warmup = 5  # 默认 3
```

### 自定义容差（tolerance）

```python
passed, msg = perf_test.test_correctness(
    inputs=(x_npu,),
    reference_inputs=(x_cpu,),
    atol=1e-3,  # 绝对容差
    rtol=1e-3   # 相对容差
)
```

### 手动计算统计

```python
times = [0.1, 0.2, 0.15, 0.18, 0.12]
stats = perf_test.calculate_statistics(times)
print(stats['statistics']['median'])  # 自动计算中位数
```

---

## 📦 BasePerformanceTest 提供的功能

| 功能 | 方法 | 说明 |
|------|------|------|
| 设备管理 | `setup_device()` | 自动选择 NPU/CUDA/CPU |
| 缓存清理 | `clear_cache()` | 清理设备缓存 |
| 同步 | `synchronize()` | 同步设备 |
| Warmup | `warmup(*inputs)` | 预热运行 |
| 性能测试 | `measure_performance(*inputs)` | 完整性能测试流程 |
| 统计计算 | `calculate_statistics(times)` | 计算所有统计指标 |
| 保存结果 | `save_results(results, file)` | JSON 格式保存 |
| 正确性测试 | `test_correctness(...)` | 自动对比输出 |

---

## 📊 自动生成的结果格式

```json
{
  "raw_data": [0.15, 0.14, 0.16, ...],
  "statistics": {
    "mean": 0.147,
    "median": 0.145,
    "std": 0.009,
    "min": 0.135,
    "max": 0.171,
    "num_trials": 20
  }
}
```

---

## 🎓 实际例子

### 1. 简单算子（Add）

```python
class AddPerformanceTest(BasePerformanceTest):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def run_operator(self, *inputs):
        return add_custom.run_add_custom(self.x, self.y)

    def run_reference(self, *inputs):
        return torch.add(self.x.cpu(), self.y.cpu())
```

### 2. 复杂算子（Matmul with Bias）

```python
class MatmulPerformanceTest(BasePerformanceTest):
    def __init__(self, a, b, bias):
        super().__init__()
        self.a = a
        self.b = b
        self.bias = bias

    def run_operator(self, *inputs):
        return matmul_custom.run_matmul_custom(self.a, self.b, self.bias)

    def run_reference(self, *inputs):
        a_ref = self.a.cpu().type(torch.float32)
        b_ref = self.b.cpu().type(torch.float32)
        return torch.matmul(a_ref, b_ref) + self.bias.cpu()
```

---

## 🐛 常见问题

### Q: 如何调整测试次数？

A: 使用 `--runs` 参数：
```bash
python evaluater.py --op your_op --runs 50
```

### Q: 缓存清理影响性能吗？

A: 会略微增加测试时间，但能保证测试公平性。缓存清理只在测试前执行，不计入性能时间。

### Q: 支持 CPU 测试吗？

A: 完全支持！`BasePerformanceTest` 会自动检测并使用合适的计时方法。

### Q: 如何添加自定义统计指标？

A: 重写 `calculate_statistics` 方法：
```python
def calculate_statistics(self, times):
    results = super().calculate_statistics(times)
    # 添加自定义指标
    results['statistics']['p95'] = sorted(times)[int(len(times)*0.95)]
    return results
```

---

## 📝 总结

使用模块化框架，你只需要：

1. ✅ 复制模板
2. ✅ 替换算子名称
3. ✅ 实现 3 个方法（存储输入、运行算子、运行参考）

**就这样！** 🎉

所有复杂的功能（缓存清理、性能测试、统计、保存）都是自动的！

---

**Happy Testing!** 🚀
