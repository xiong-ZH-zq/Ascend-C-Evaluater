#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from perf_test_base import BasePerformanceTest
from matmul_input_config import MatmulInputConfig
from matmul_standard import Model as StandardModel

sys.path.append(os.getcwd())
import matmul_custom


class MatmulPerformanceTest(BasePerformanceTest):
    """Performance test wrapper for matmul custom operator"""

    def __init__(self, input_config, standard_model):
        """
        Initialize with input config and standard model.

        Args:
            input_config: MatmulInputConfig instance
            standard_model: Standard PyTorch model for reference
        """
        super().__init__()
        self.input_config = input_config
        self.standard_model = standard_model
        self.inputs_npu = None  # Will be set when device is ready

    def prepare_inputs(self):
        """Prepare inputs on NPU device"""
        if self.inputs_npu is None:
            self.inputs_npu = self.input_config.get_inputs_for_device('npu:0')
        return self.inputs_npu

    def run_operator(self, *inputs):
        """Run custom matmul operator"""
        if not inputs:
            inputs = self.prepare_inputs()
        return matmul_custom.run_matmul_custom(*inputs)

    def run_reference(self, *inputs):
        """Run reference PyTorch implementation"""
        if not inputs:
            # Get CPU inputs
            inputs_cpu = self.input_config.get_inputs_for_device('cpu')
        else:
            # Move to CPU if needed
            inputs_cpu = [inp.cpu() if hasattr(inp, 'cpu') else inp for inp in inputs]

        # Convert float16 to float32 for CPU compatibility
        inputs_cpu_converted = []
        for inp in inputs_cpu:
            if isinstance(inp, torch.Tensor) and inp.dtype == torch.float16:
                inputs_cpu_converted.append(inp.float())
            else:
                inputs_cpu_converted.append(inp)

        # Use standard model for reference
        with torch.no_grad():
            output = self.standard_model(*inputs_cpu_converted)

        return output


class TestCustomMatmul(TestCase):

    def test_matmul_custom_ops(self):
        """Test correctness and measure performance of custom matmul operator"""

        # ===================================================================
        # Step 1: Create input configuration (define once, use everywhere)
        # ===================================================================
        input_config = MatmulInputConfig(M=1024, K=256, N=640, seed=1024)
        print(input_config)  # Print input shapes

        # ===================================================================
        # Step 2: Create standard model (define once, use for both tests)
        # ===================================================================
        standard_model = StandardModel()

        # ===================================================================
        # Step 3: Prepare inputs on NPU
        # ===================================================================
        inputs_npu = input_config.get_inputs_for_device('npu:0')
        a, b, bias = inputs_npu

        # ===================================================================
        # Step 4: Correctness test (First use of standard model)
        # ===================================================================
        print("[INFO] Testing correctness...")
        perf_test = MatmulPerformanceTest(input_config, standard_model)
        perf_test.setup_device()

        custom_output = perf_test.run_operator(*inputs_npu)
        reference_output = perf_test.run_reference(*inputs_npu)

        # Type conversion for comparison
        if custom_output.dtype != reference_output.dtype:
            reference_output = reference_output.type(custom_output.dtype)

        self.assertRtolEqual(custom_output, reference_output)
        print("[PASS] Correctness test passed")

        # ===================================================================
        # Step 5: Performance test (Second use of standard model in evaluater.py)
        # ===================================================================
        results = perf_test.measure_performance()
        perf_test.save_results(results)


if __name__ == "__main__":
    run_tests()
