#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Testing Base Module

This module provides common functionality for operator performance testing,
including cache clearing, timing, and statistics calculation.

Usage:
    from perf_test_base import BasePerformanceTest

    class TestCustomAdd(BasePerformanceTest):
        def run_operator(self, *inputs):
            return add_custom.run_add_custom(*inputs)

        def run_reference(self, *inputs):
            return torch.add(*inputs)
"""

import os
import json
import torch

try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    torch_npu = None


class BasePerformanceTest:
    """
    Base class for performance testing with common utilities.

    Features:
    - Automatic cache clearing
    - Performance measurement with statistics
    - Progress reporting
    - JSON result export
    """

    def __init__(self):
        self.num_trials = int(os.environ.get('PERF_NUM_TRIALS', '100'))
        self.num_warmup = 3
        self.device = None

    def setup_device(self):
        """Setup NPU/CUDA/CPU device"""
        if NPU_AVAILABLE and torch.npu.is_available():
            self.device = torch.device('npu:0')
            torch.npu.config.allow_internal_format = False
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        return self.device

    def clear_cache(self):
        """Clear device cache before measurement"""
        if self.device is None:
            return

        if self.device.type == 'npu' and NPU_AVAILABLE:
            torch_npu.npu.empty_cache()
            torch_npu.npu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def synchronize(self):
        """Synchronize device"""
        if self.device is None:
            return

        if self.device.type == 'npu' and NPU_AVAILABLE:
            torch_npu.npu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()

    def run_operator(self, *inputs):
        """
        Override this method to run your custom operator.

        Args:
            *inputs: Operator inputs

        Returns:
            Output tensor
        """
        raise NotImplementedError("Must implement run_operator()")

    def run_reference(self, *inputs):
        """
        Override this method to run reference implementation.

        Args:
            *inputs: Operator inputs

        Returns:
            Output tensor
        """
        raise NotImplementedError("Must implement run_reference()")

    def warmup(self, *inputs):
        """Warmup runs before measurement"""
        print(f"[INFO] Warming up ({self.num_warmup} iterations)...")
        for _ in range(self.num_warmup):
            _ = self.run_operator(*inputs)
            self.synchronize()

    def measure_performance(self, *inputs):
        """
        Measure operator performance with cache clearing.

        Args:
            *inputs: Operator inputs

        Returns:
            dict: Performance results with raw_data and statistics
        """
        print(f"[INFO] Running performance test with {self.num_trials} trials")

        # Warmup
        self.warmup(*inputs)

        # Measure
        execution_times = []

        if self.device.type == 'npu' and NPU_AVAILABLE:
            for i in range(self.num_trials):
                # Clear cache before each measurement
                self.clear_cache()

                start_event = torch_npu.npu.Event(enable_timing=True)
                end_event = torch_npu.npu.Event(enable_timing=True)

                start_event.record()
                _ = self.run_operator(*inputs)
                end_event.record()

                torch_npu.npu.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                execution_times.append(elapsed_time_ms)

                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == self.num_trials:
                    print(f"[PERF] Completed {i + 1}/{self.num_trials} trials")

        elif self.device.type == 'cuda':
            for i in range(self.num_trials):
                # Clear cache before each measurement
                self.clear_cache()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = self.run_operator(*inputs)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                execution_times.append(elapsed_time_ms)

                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == self.num_trials:
                    print(f"[PERF] Completed {i + 1}/{self.num_trials} trials")

        else:
            # CPU timing
            import time
            for i in range(self.num_trials):
                start_time = time.time()
                _ = self.run_operator(*inputs)
                end_time = time.time()
                execution_times.append((end_time - start_time) * 1000)

                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == self.num_trials:
                    print(f"[PERF] Completed {i + 1}/{self.num_trials} trials")

        # Calculate statistics
        return self.calculate_statistics(execution_times)

    def calculate_statistics(self, execution_times):
        """
        Calculate statistics from execution times.

        Args:
            execution_times: List of execution times in milliseconds

        Returns:
            dict: Results with raw_data and statistics
        """
        execution_times_sorted = sorted(execution_times)
        mean_time = sum(execution_times) / len(execution_times)

        # Calculate median
        n = len(execution_times_sorted)
        if n % 2 == 0:
            median_time = (execution_times_sorted[n//2 - 1] + execution_times_sorted[n//2]) / 2
        else:
            median_time = execution_times_sorted[n//2]

        # Calculate standard deviation
        variance = sum((x - mean_time)**2 for x in execution_times) / len(execution_times)
        std_time = variance ** 0.5

        results = {
            'raw_data': execution_times,
            'statistics': {
                'mean': mean_time,
                'median': median_time,
                'min': min(execution_times),
                'max': max(execution_times),
                'std': std_time,
                'num_trials': self.num_trials
            }
        }

        return results

    def save_results(self, results, filename='timing_results.json'):
        """
        Save timing results to JSON file.

        Args:
            results: Performance results dictionary
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        stats = results.get('statistics', {})
        print(f"[PERF] Mean execution time: {stats.get('mean', 0):.3f} ms")
        print(f"[PERF] Median execution time: {stats.get('median', 0):.3f} ms")

    def test_correctness(self, inputs, reference_inputs=None, atol=1e-2, rtol=1e-2):
        """
        Test correctness by comparing operator output with reference.

        Args:
            inputs: Inputs for custom operator
            reference_inputs: Inputs for reference (if different from inputs)
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            tuple: (bool: passed, str: message)
        """
        if reference_inputs is None:
            reference_inputs = inputs

        custom_output = self.run_operator(*inputs)
        reference_output = self.run_reference(*reference_inputs)

        # Move to CPU for comparison if needed
        if hasattr(custom_output, 'cpu'):
            custom_output = custom_output.cpu()
        if hasattr(reference_output, 'cpu'):
            reference_output = reference_output.cpu()

        # Convert to same dtype if needed
        if custom_output.dtype != reference_output.dtype:
            reference_output = reference_output.type(custom_output.dtype)

        try:
            if torch.allclose(custom_output, reference_output, atol=atol, rtol=rtol):
                max_diff = torch.max(torch.abs(custom_output - reference_output)).item()
                return True, f"Outputs match (max diff: {max_diff:.6f})"
            else:
                max_diff = torch.max(torch.abs(custom_output - reference_output)).item()
                mean_diff = torch.mean(torch.abs(custom_output - reference_output)).item()
                return False, f"Output mismatch (max: {max_diff:.6f}, mean: {mean_diff:.6f})"
        except Exception as e:
            return False, f"Comparison error: {str(e)}"
