# Ascend C Evaluator

## Introduction

This project is a modular evaluation framework for Ascend C operators, similar to the evaluation part in [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench).

### Key Features

- ✅ **Automated Cache Clearing**: Ensures fair performance measurements
- ✅ **Modular Architecture**: Separate input configuration, operator logic, and testing
- ✅ **Comprehensive Evaluation**: Tests compilation, correctness, and performance
- ✅ **Minimal Code**: Add new operators with minimal boilerplate

## Environment Setup

### Prerequisites

1. **Ascend CANN Toolkit**: Install CANN toolkit 8.2.RC1 or compatible version
   - Download from [Ascend Community](https://www.hiascend.com/software/cann)
   - Follow official installation guide for your system

2. **Python 3.9+**: Recommended Python 3.9 or higher

3. **CMake**: Version 3.16.0 or higher for building operators

### Installation Steps

#### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n evaluater python=3.10
conda activate evaluater
```

#### 2. Install PyTorch and torch_npu

```bash
# Install PyTorch 2.1.0
pip install torch==2.1.0

# Install torch_npu (ensure it matches your CANN version)
# The torch_npu wheel is typically provided with CANN toolkit
# Example for CANN 8.2.RC1:
pip install torch_npu==2.1.0.post12
```

**Note**: `torch_npu` version must be compatible with both PyTorch and CANN toolkit versions. Refer to the [torch_npu compatibility matrix](https://gitee.com/ascend/pytorch).

#### 3. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt
```

#### 4. Set Environment Variables

Add CANN toolkit paths to your environment:

```bash
# Add to ~/.bashrc or ~/.zshrc
export ASCEND_TOOLKIT_HOME=/path/to/Ascend/ascend-toolkit/8.2.RC1
source ${ASCEND_TOOLKIT_HOME}/set_env.sh

# Verify installation
npu-smi info  # Should show NPU device information
```

#### 5. Verify Installation

```bash
# Test torch_npu installation
python -c "import torch; import torch_npu; print(f'NPU available: {torch.npu.is_available()}')"

# Should output: NPU available: True
```

### Quick Environment Setup Script

```bash
#!/bin/bash
# setup_env.sh

# Create and activate conda environment
conda create -n evaluater python=3.8 -y
conda activate evaluater

# Install PyTorch and torch_npu
pip install torch==2.1.0
pip install torch_npu==2.1.0.post12

# Install dependencies
pip install -r requirements.txt

# Set CANN environment (adjust path as needed)
export ASCEND_TOOLKIT_HOME=/data1/Ascend/ascend-toolkit/8.2.RC1
source ${ASCEND_TOOLKIT_HOME}/set_env.sh

echo "Environment setup complete!"
echo "Activate with: conda activate evaluater"
```

### Troubleshooting

**Issue**: `ImportError: cannot import name 'torch_npu'`
- **Solution**: Ensure torch_npu is installed and compatible with your PyTorch version

**Issue**: `NPU device not found`
- **Solution**: Check CANN toolkit installation and environment variables
- Run `npu-smi info` to verify NPU is recognized

**Issue**: CMake build errors
- **Solution**: Ensure CMake 3.16+ is installed: `cmake --version`
- Install with: `conda install cmake` or `pip install cmake`

## Project Structure

```bash
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── dataset.py                   # (Optional) Operator registry - not required
├── evaluater.py                 # Main evaluation script
├── perf_test_base.py            # Performance testing base class
├── input_config_base.py         # Input configuration base class
├── utils.py                     # Utility functions
├── QUICK_START.md               # Quick start guide
├── TEMPLATE_custom_test.py      # Template for new operators
│
└── op/                          # Operator implementations
    ├── add/
    │   ├── add_custom.cpp           # Custom Ascend C implementation
    │   ├── add_custom_test.py       # Test file with performance measurement
    │   ├── add_input_config.py      # Input data configuration
    │   ├── add_standard.py          # Standard PyTorch implementation
    │   ├── CMakeLists.txt           # Build configuration
    │   ├── pybind11.cpp             # Python bindings
    │   └── run.sh                   # Build script
    │
    └── matmul/
        ├── matmul_custom.cpp        # Custom implementation
        ├── matmul_custom_tiling.cpp # Tiling optimization
        ├── matmul_custom_test.py    # Test file
        ├── matmul_input_config.py   # Input configuration
        ├── matmul_standard.py       # Standard implementation
        ├── CMakeLists.txt
        ├── pybind11.cpp
        └── run.sh
```

## Usage

### Basic Command

Run evaluation for an operator:

```shell
conda activate evaluater
python evaluater.py --op matmul --runs 10
```

**Parameters:**
- `--op`: Name of the operator to evaluate
- `--runs`: Number of trials for performance measurement (default: 1)
- `--output`: Custom output file path (default: `result_{op}.json`)

### Evaluation Workflow

The evaluator performs three essential checks:

1. **Compilation**: Build the custom operator using CMake
2. **Correctness**: Verify output matches standard PyTorch implementation
3. **Performance**: Measure speedup compared to standard PyTorch

Example output:
```
[STEP 1] Building operator...                   ✓ PASS
[STEP 2] Loading standard implementation...     ✓ PASS
[STEP 3] Running custom operator test...        ✓ PASS
[STEP 4] Testing correctness...                 ✓ PASS
[STEP 5] Measuring standard performance...      ✓ PASS
[STEP 6] Calculating speedup...                 ✓ 0.42x

Results saved to: result_matmul.json
```

> If you use this project for LLM GPU Kernel generation, let LLM generate `{op}_custom.cpp` and `{op}_custom_tiling.cpp`.

## Adding a New Operator

This section demonstrates how to add a new operator (using **ReLU** as an example) to the evaluation framework.

### Architecture Overview

The framework uses a **three-layer separation** design:

```
Layer 1: Input Configuration  → relu_input_config.py
Layer 2: Standard Model       → relu_standard.py
Layer 3: Custom Test          → relu_custom_test.py
```

This ensures:
- Input data is defined **once** and reused everywhere
- Standard PyTorch model is defined **once** and used for both correctness and performance testing
- No code duplication

---

### Step-by-Step Guide: Adding ReLU Operator

#### Step 1: Create Operator Directory

```bash
mkdir -p op/relu
cd op/relu
```

#### Step 2: Create Input Configuration

Create `op/relu/relu_input_config.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReLU Operator Input Configuration
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from input_config_base import BaseInputConfig


class ReluInputConfig(BaseInputConfig):
    """Input configuration for ReLU operator"""

    def __init__(self, shape=(1024, 1024), seed=None):
        """
        Initialize ReLU input configuration.

        Args:
            shape: Shape of input tensor
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.shape = shape

    def generate_inputs(self):
        """
        Generate inputs for ReLU: [x]

        Returns:
            list: [x(shape)]
        """
        x = torch.randn(*self.shape, dtype=torch.float16)
        return [x]

    def get_input_shapes(self):
        """Return input shapes without generating tensors"""
        return [self.shape]
```

**What this does:**
- Defines the shape and dtype of inputs
- Supports customizable parameters (shape, seed)
- Inherits device conversion methods from `BaseInputConfig`

#### Step 3: Create Standard PyTorch Implementation

Create `op/relu/relu_standard.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard PyTorch Implementation for ReLU Operator
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard PyTorch ReLU operator

    Used for:
    1. Correctness verification (in custom test)
    2. Performance baseline (in evaluator)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor (same shape as input)
        """
        return torch.relu(x)


# Backward compatibility functions
def get_init_inputs():
    """No initialization inputs needed"""
    return []


def get_inputs():
    """Generate inputs using input config"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from relu_input_config import ReluInputConfig
    config = ReluInputConfig()
    return config.generate_inputs()
```

**What this does:**
- Defines **only** the operator logic (ReLU activation)
- No input data generation (that's in `relu_input_config.py`)
- Reusable for both correctness and performance testing

#### Step 4: Create Custom Test File

Create `op/relu/relu_custom_test.py`:

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from perf_test_base import BasePerformanceTest
from relu_input_config import ReluInputConfig
from relu_standard import Model as StandardModel

sys.path.append(os.getcwd())
import relu_custom  # Your custom pybind11 module


class ReluPerformanceTest(BasePerformanceTest):
    """Performance test wrapper for ReLU custom operator"""

    def __init__(self, input_config, standard_model):
        super().__init__()
        self.input_config = input_config
        self.standard_model = standard_model
        self.inputs_npu = None

    def prepare_inputs(self):
        """Prepare inputs on NPU device"""
        if self.inputs_npu is None:
            self.inputs_npu = self.input_config.get_inputs_for_device('npu:0')
        return self.inputs_npu

    def run_operator(self, *inputs):
        """Run custom ReLU operator"""
        if not inputs:
            inputs = self.prepare_inputs()
        return relu_custom.run_relu_custom(*inputs)

    def run_reference(self, *inputs):
        """Run standard PyTorch ReLU"""
        if not inputs:
            inputs_cpu = self.input_config.get_inputs_for_device('cpu')
        else:
            inputs_cpu = [inp.cpu() if hasattr(inp, 'cpu') else inp for inp in inputs]

        # Convert float16 to float32 for CPU compatibility
        inputs_cpu_converted = []
        for inp in inputs_cpu:
            if isinstance(inp, torch.Tensor) and inp.dtype == torch.float16:
                inputs_cpu_converted.append(inp.float())
            else:
                inputs_cpu_converted.append(inp)

        with torch.no_grad():
            output = self.standard_model(*inputs_cpu_converted)

        return output


class TestCustomRelu(TestCase):

    def test_relu_custom_ops(self):
        """Test correctness and measure performance"""

        # Step 1: Create input configuration (define once, use everywhere)
        input_config = ReluInputConfig(shape=(1024, 1024), seed=1024)
        print(input_config)

        # Step 2: Create standard model (define once, use twice)
        standard_model = StandardModel()

        # Step 3: Prepare inputs on NPU
        inputs_npu = input_config.get_inputs_for_device('npu:0')
        x = inputs_npu[0]

        # Step 4: Correctness test
        print("[INFO] Testing correctness...")
        perf_test = ReluPerformanceTest(input_config, standard_model)
        perf_test.setup_device()

        custom_output = perf_test.run_operator(*inputs_npu)
        reference_output = perf_test.run_reference(*inputs_npu)

        # Type conversion for comparison
        if custom_output.dtype != reference_output.dtype:
            reference_output = reference_output.type(custom_output.dtype)

        self.assertRtolEqual(custom_output, reference_output)
        print("[PASS] Correctness test passed")

        # Step 5: Performance test
        results = perf_test.measure_performance()
        perf_test.save_results(results)


if __name__ == "__main__":
    run_tests()
```

**What this does:**
- Uses `ReluInputConfig` for input generation
- Uses `StandardModel` for correctness verification
- Automatically handles cache clearing and performance measurement
- All complex logic is inherited from `BasePerformanceTest`

#### Step 5: Create Custom Operator Files

You need to provide these files (typically generated by LLM or written manually):

**`op/relu/relu_custom.cpp`** - Your Ascend C implementation

**`op/relu/pybind11.cpp`** - Python bindings:
```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>
// Your kernel header
#include "relu_custom.h"

torch::Tensor run_relu_custom(torch::Tensor x) {
    // Call your Ascend C kernel
    return relu_kernel(x);
}

PYBIND11_MODULE(relu_custom, m) {
    m.def("run_relu_custom", &run_relu_custom, "ReLU custom operator");
}
```

**`op/relu/CMakeLists.txt`** - Build configuration (copy from add/matmul and modify)

#### Step 6: Run Evaluation

```bash
# Navigate to project root
cd /path/to/Ascend-C-Evaluater

# Run evaluation
python evaluater.py --op relu --runs 20
```

**Expected output:**
```
================================================================================
Evaluating operator: relu
Runs: 20
================================================================================

[STEP 1] Building operator from op/relu
[INFO] Building operator...
[PASS] Compilation succeeded

[STEP 2] Loading standard PyTorch implementation: relu_standard.py
[PASS] Standard implementation loaded successfully

[STEP 3] Running custom operator test
Input Configuration:
  Input 0: shape=(1024, 1024), dtype=torch.float16
[INFO] Testing correctness...
[PASS] Correctness test passed
[INFO] Running performance test with 20 trials
[PERF] Completed 10/20 trials
[PERF] Completed 20/20 trials
[PERF] Mean execution time: 0.145 ms
[PERF] Median execution time: 0.142 ms
[INFO] Custom operator timing:
       Mean: 0.145 ms
       Median: 0.142 ms
[PASS] C++ test passed

[STEP 4] Running PyTorch reference for comparison
[INFO] Using NPU device: npu:0
[PASS] Standard model initialized

[STEP 5] Measuring standard PyTorch performance (20 trials)
[PERF] Standard PyTorch: Completed 10/20 trials
[PERF] Standard PyTorch: Completed 20/20 trials
[INFO] Standard PyTorch performance:
  Mean:   0.138 ms
  Median: 0.136 ms
  Std:    0.012 ms
  Min:    0.121 ms
  Max:    0.165 ms

[INFO] Speedup (based on mean):   0.95x
[INFO] Speedup (based on median): 0.96x

[STEP 6] Correctness validation
[PASS] Correctness validation passed

================================================================================
Evaluation Summary:
  Compiled: True
  Correctness: True
  Speedup (mean):   0.95x
  Speedup (median): 0.96x
================================================================================

[INFO] Results saved to: result_relu.json
```

#### Step 7: Check Results

The results are saved in `result_relu.json`:

```json
{
  "operator": "relu",
  "compiled": true,
  "correctness": true,
  "performance": {
    "standard": {
      "raw_data": [0.138, 0.142, ...],
      "statistics": {
        "mean": 0.138,
        "median": 0.136,
        "std": 0.012,
        "min": 0.121,
        "max": 0.165,
        "num_trials": 20
      }
    },
    "custom": {
      "raw_data": [0.145, 0.148, ...],
      "statistics": {
        "mean": 0.145,
        "median": 0.142,
        "std": 0.011,
        "min": 0.133,
        "max": 0.167,
        "num_trials": 20
      }
    }
  },
  "speedup": {
    "mean": 0.95,
    "median": 0.96
  },
  "compile_info": "",
  "correctness_info": "C++ test passed and reference runs successfully",
  "performance_info": ""
}
```

---

### Summary: What You Need to Create

For each new operator, you only need to create **3 simple files**:

| File | Lines of Code | Purpose |
|------|--------------|---------|
| `{op}_input_config.py` | ~30 lines | Define input data |
| `{op}_standard.py` | ~40 lines | Define operator logic |
| `{op}_custom_test.py` | ~80 lines | Wire everything together |
| **Total** | **~150 lines** | **vs. 300+ lines before!** |

**Everything else is automatic:**
- ✅ Cache clearing
- ✅ Performance timing
- ✅ Statistics calculation (mean, median, std, min, max)
- ✅ Progress reporting
- ✅ JSON export
- ✅ Device management

**Note:** You do **NOT** need to update `dataset.py` when adding new operators. The evaluator automatically discovers operators by checking the `op/` directory.

## Evaluation Components

Each operator evaluation includes three essential components:

1. **Compilation**: Verifies that the custom Ascend C operator can be built successfully
2. **Correctness**: Validates that the custom operator produces correct results (compared against standard PyTorch)
3. **Speedup**: Measures performance improvement over standard PyTorch implementation
   - Speedup > 1.0x: Custom operator is faster ✓
   - Speedup < 1.0x: Custom operator is slower (needs optimization)

## Advanced Features

### Custom Input Configurations

You can customize input parameters:

```python
# Different sizes
input_config = MatmulInputConfig(M=2048, K=512, N=1024)

# Different random seed
input_config = MatmulInputConfig(seed=42)
```

### Custom Number of Trials

```bash
# More trials for better statistics
python evaluater.py --op relu --runs 100

# Fewer trials for quick testing
python evaluater.py --op relu --runs 5
```

### Custom Output Path

```bash
python evaluater.py --op relu --runs 20 --output my_results.json
```

## Frequently Asked Questions

### Do I need to update `dataset.py` when adding a new operator?

**No!** The `dataset.py` file is **optional** and not used by the current evaluator. It's a legacy file kept for potential future use (e.g., batch evaluation, operator categorization).

The evaluator automatically discovers operators by checking if the directory `op/{operator_name}/` exists. You only need to:
1. Create the operator directory
2. Add the 3 required files (input_config, standard, custom_test)
3. Run: `python evaluater.py --op {operator_name}`

### What files are actually required for a new operator?

**Required files:**
```
op/{op_name}/
├── {op}_input_config.py     ✅ Required
├── {op}_standard.py          ✅ Required
├── {op}_custom_test.py       ✅ Required
├── {op}_custom.cpp           ✅ Required (your implementation)
├── pybind11.cpp              ✅ Required
└── CMakeLists.txt            ✅ Required
```

**NOT required:**
- ❌ Updating `dataset.py`
- ❌ Updating `config.py`
- ❌ Updating `evaluater.py`

### How does the evaluator find my operator?

Simple file system check:
```python
# When you run: python evaluater.py --op relu
op_path = f"op/{args.op}"  # -> "op/relu"
if os.path.exists(op_path):
    # Found it! Proceed with evaluation
```

## Troubleshooting

### Build Errors

If you encounter CMake errors, try cleaning the build directory:

```bash
rm -rf op/relu/build
python evaluater.py --op relu --runs 10
```

### Import Errors

Make sure you're running from the project root:

```bash
cd /path/to/Ascend-C-Evaluater
python evaluater.py --op relu --runs 10
```

### Correctness Failures

Check that your custom operator implementation matches the PyTorch reference:
1. Verify input/output shapes
2. Check data types
3. Ensure proper device placement (NPU vs CPU)

## Performance Tips

1. **Cache Clearing**: Enabled by default to ensure fair measurements
2. **Warmup Runs**: 3 warmup iterations before actual measurement
3. **Multiple Trials**: Use `--runs 20` or more for reliable statistics
4. **Median vs Mean**: Median is more robust to outliers

## References

- [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) - Original inspiration
- [Ascend C Documentation](https://www.hiascend.com/en/software/cann) - Ascend C programming guide
- [QUICK_START.md](QUICK_START.md) - Detailed quick start guide

## License

This project follows the original licensing terms. See individual source files for details.
