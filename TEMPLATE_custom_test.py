#!/usr/bin/python3
# coding=utf-8
"""
Template for creating new custom operator tests.

Usage:
1. Copy this file to your operator directory: op/{operator_name}/{operator_name}_custom_test.py
2. Replace {OPERATOR_NAME} with your operator name (e.g., "conv2d", "relu")
3. Update the input preparation section
4. Implement run_operator() and run_reference() methods
5. Done! The performance testing, cache clearing, and statistics are automatic.

Example:
    For a "relu" operator:
    - File: op/relu/relu_custom_test.py
    - Class: ReluPerformanceTest
    - Import: import relu_custom
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from perf_test_base import BasePerformanceTest

sys.path.append(os.getcwd())
# TODO: Replace with your operator module
import {OPERATOR_NAME}_custom


class {OPERATOR_NAME_CAPITALIZED}PerformanceTest(BasePerformanceTest):
    """Performance test wrapper for {OPERATOR_NAME} custom operator"""

    def __init__(self, *inputs):
        """
        Initialize with operator inputs.

        Args:
            *inputs: All input tensors needed for your operator

        Example:
            For matmul: __init__(self, a, b, bias)
            For add: __init__(self, x, y)
        """
        super().__init__()
        # TODO: Store your inputs
        # Example:
        # self.input1 = inputs[0]
        # self.input2 = inputs[1]
        self.inputs = inputs

    def run_operator(self, *inputs):
        """
        Run custom operator.

        Override this to call your custom operator function.

        Returns:
            Output tensor from your custom operator
        """
        # TODO: Replace with your custom operator call
        # Example for matmul:
        # return matmul_custom.run_matmul_custom(self.a, self.b, self.bias)

        if inputs:
            return {OPERATOR_NAME}_custom.run_{OPERATOR_NAME}_custom(*inputs)
        return {OPERATOR_NAME}_custom.run_{OPERATOR_NAME}_custom(*self.inputs)

    def run_reference(self, *inputs):
        """
        Run reference PyTorch implementation.

        Override this to call the standard PyTorch equivalent.
        This is used for correctness checking.

        Returns:
            Output tensor from PyTorch reference
        """
        # TODO: Replace with PyTorch reference implementation
        # Example for matmul:
        # a_ref, b_ref, bias_ref = inputs if inputs else (self.a.cpu(), self.b.cpu(), self.bias.cpu())
        # return torch.matmul(a_ref, b_ref) + bias_ref

        if inputs:
            ref_inputs = inputs
        else:
            # Move inputs to CPU for reference
            ref_inputs = [inp.cpu() if hasattr(inp, 'cpu') else inp for inp in self.inputs]

        # TODO: Call PyTorch reference function
        # Example: return torch.add(*ref_inputs)
        pass


class TestCustom{OPERATOR_NAME_CAPITALIZED}(TestCase):
    """Test case for {OPERATOR_NAME} custom operator"""

    def test_{OPERATOR_NAME}_custom_ops(self):
        """Test correctness and measure performance of custom {OPERATOR_NAME} operator"""

        # ====================================================================
        # TODO: Prepare inputs for your operator
        # ====================================================================
        # Example for matmul:
        # a = torch.rand([1024, 256], device='cpu', dtype=torch.float16).npu()
        # b = torch.rand([256, 640], device='cpu', dtype=torch.float16).npu()
        # bias = torch.randn([640], device='cpu', dtype=torch.float32).npu()
        # inputs = (a, b, bias)

        # Example for add:
        # x = torch.rand([8, 4096], device='cpu', dtype=torch.float16).npu()
        # y = torch.rand([8, 4096], device='cpu', dtype=torch.float16).npu()
        # inputs = (x, y)

        inputs = None  # TODO: Replace with your actual inputs
        # ====================================================================

        # Create performance test instance
        perf_test = {OPERATOR_NAME_CAPITALIZED}PerformanceTest(*inputs)
        perf_test.setup_device()

        # Correctness test
        print("[INFO] Testing correctness...")
        output = perf_test.run_operator()

        # TODO: Prepare reference output
        # Example for matmul:
        # cpuout = torch.matmul(a.cpu().type(output.dtype), b.cpu().type(output.dtype)) + bias.cpu()

        # Example for add:
        # cpuout = torch.add(x.cpu(), y.cpu())

        cpuout = perf_test.run_reference()  # Or compute manually as above

        self.assertRtolEqual(output, cpuout)
        print("[PASS] Correctness test passed")

        # Performance measurement (automatic with cache clearing)
        results = perf_test.measure_performance()

        # Save results (automatic JSON export)
        perf_test.save_results(results)


if __name__ == "__main__":
    run_tests()


# ============================================================================
# Quick Start Guide
# ============================================================================
"""
STEP 1: Copy this template
---------------------------
cp TEMPLATE_custom_test.py op/your_op/your_op_custom_test.py

STEP 2: Find and replace
-------------------------
- Replace {OPERATOR_NAME} with "your_op" (e.g., "conv2d", "relu")
- Replace {OPERATOR_NAME_CAPITALIZED} with "YourOp" (e.g., "Conv2d", "Relu")

STEP 3: Implement your operator
--------------------------------
In run_operator():
    return your_op_custom.run_your_op_custom(*self.inputs)

In run_reference():
    return torch.your_op(*ref_inputs)

STEP 4: Prepare inputs
-----------------------
In test_your_op_custom_ops():
    x = torch.rand([shape], dtype=...).npu()
    inputs = (x, ...)

STEP 5: Run!
------------
The base module automatically handles:
✓ Cache clearing before each measurement
✓ Performance timing with progress indicators
✓ Statistics calculation (mean, median, std, min, max)
✓ JSON result export
✓ Warmup runs

You get all this for FREE! Just implement the 3 methods above.
"""
