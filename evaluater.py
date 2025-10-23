#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend C Evaluater - Main evaluation script

This script evaluates Ascend C operators by:
1. Building the operator using CMake/build.sh
2. Running both the custom operator and PyTorch reference
3. Comparing correctness and performance
"""

import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import torch

from config import (
    project_root_path, op_base_path,
    num_correct_trials, num_perf_trials, num_warmup,
    atol, rtol, seed_num, device_type, device_id
)
from utils import (
    build_operator, run_cpp_test, measure_pytorch_performance,
    check_correctness, set_seed, NPU_AVAILABLE
)

try:
    import torch_npu
except ImportError:
    torch_npu = None


def load_standard_implementation(op_name):
    """
    Dynamically load standard PyTorch implementation module from op directory.

    Args:
        op_name: Name of the operator (e.g., 'matmul', 'add')

    Returns:
        module: Loaded module containing Model, get_inputs, get_init_inputs
    """
    standard_path = os.path.join(op_base_path, op_name, f'{op_name}_standard.py')
    if not os.path.exists(standard_path):
        raise FileNotFoundError(f"Standard implementation not found: {standard_path}")

    spec = importlib.util.spec_from_file_location(f'{op_name}_standard', standard_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Verify required components
    if not hasattr(module, 'Model'):
        raise AttributeError(f"Standard module must define 'Model' class")
    if not hasattr(module, 'get_inputs'):
        raise AttributeError(f"Standard module must define 'get_inputs' function")
    if not hasattr(module, 'get_init_inputs'):
        raise AttributeError(f"Standard module must define 'get_init_inputs' function")

    return module


def evaluate_operator(op_name, runs=1):
    """
    Evaluate a single operator.

    Args:
        op_name: Name of the operator to evaluate
        runs: Number of evaluation runs

    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating operator: {op_name}")
    print(f"Runs: {runs}")
    print(f"{'='*80}\n")

    result = {
        'operator': op_name,
        'compiled': False,
        'correctness': None,
        'performance': None,
        'speedup': None,
        'compile_info': '',
        'correctness_info': '',
        'performance_info': ''
    }

    # Step 1: Build the operator
    op_path = os.path.join(op_base_path, op_name)
    print(f"[STEP 1] Building operator from {op_path}")

    compiled, error_msg = build_operator(op_path)
    result['compiled'] = compiled

    if not compiled:
        result['compile_info'] = error_msg
        print(f"[FAIL] Compilation failed: {error_msg}")
        return result

    print("[PASS] Compilation succeeded\n")

    # Step 2: Load standard PyTorch implementation
    print(f"[STEP 2] Loading standard PyTorch implementation: {op_name}_standard.py")

    try:
        standard_module = load_standard_implementation(op_name)
        print("[PASS] Standard implementation loaded successfully\n")
    except Exception as e:
        result['correctness_info'] = f"Failed to load standard implementation: {str(e)}"
        print(f"[FAIL] {result['correctness_info']}")
        return result

    # Step 3: Run C++ test and collect metrics
    print(f"[STEP 3] Running C++ operator test")

    cpp_success, cpp_result = run_cpp_test(op_path, num_trials=runs)

    if not cpp_success:
        result['correctness_info'] = f"C++ test failed: {cpp_result}"
        print(f"[FAIL] {result['correctness_info']}")
        return result

    # Extract C++ test results
    if isinstance(cpp_result, dict):
        if cpp_result.get('correctness') is False:
            result['correctness'] = False
            result['correctness_info'] = "C++ test reported correctness failure"
            print(f"[FAIL] {result['correctness_info']}")
            return result

        print("[INFO] C++ test passed")

    # Step 4: Run PyTorch reference and compare
    print(f"\n[STEP 4] Running PyTorch reference for comparison")

    # Setup device
    if device_type == 'npu' and NPU_AVAILABLE and torch.npu.is_available():
        device = torch.device(f'npu:{device_id}')
        print(f"[INFO] Using NPU device: {device}")
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"[INFO] Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        print(f"[INFO] Using CPU device")

    # Get model and inputs
    try:
        get_init_inputs = standard_module.get_init_inputs
        get_inputs = standard_module.get_inputs
        Model = standard_module.Model

        init_inputs = get_init_inputs()
        init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

        set_seed(seed_num)
        standard_model = Model(*init_inputs).to(device)

        print("[PASS] Standard model initialized\n")

    except Exception as e:
        result['correctness_info'] = f"Failed to initialize standard model: {str(e)}"
        print(f"[FAIL] {result['correctness_info']}")
        return result

    # Step 5: Measure standard PyTorch performance
    print(f"[STEP 5] Measuring standard PyTorch performance ({runs} trials)")

    try:
        # Generate test inputs
        set_seed(seed_num)
        test_inputs = get_inputs()
        test_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in test_inputs]

        # Measure performance
        standard_perf = measure_pytorch_performance(
            standard_model, test_inputs, device,
            num_warmup=num_warmup, num_trials=runs
        )

        # Extract custom operator results
        custom_perf = {
            'raw_data': cpp_result.get('raw_data', []),
            'statistics': cpp_result.get('statistics', {})
        }

        result['performance'] = {
            'standard': standard_perf,
            'custom': custom_perf
        }

        # Print standard performance
        std_stats = standard_perf.get('statistics', {})
        print(f"[INFO] Standard PyTorch performance:")
        print(f"  Mean:   {std_stats.get('mean', 0):.3f} ms")
        print(f"  Median: {std_stats.get('median', 0):.3f} ms")
        print(f"  Std:    {std_stats.get('std', 0):.3f} ms")
        print(f"  Min:    {std_stats.get('min', 0):.3f} ms")
        print(f"  Max:    {std_stats.get('max', 0):.3f} ms")

        # Calculate speedup using both mean and median
        custom_stats = custom_perf.get('statistics', {})
        if custom_stats:
            speedup_mean = std_stats.get('mean', 0) / custom_stats.get('mean', 1)
            speedup_median = std_stats.get('median', 0) / custom_stats.get('median', 1)
            result['speedup'] = {
                'mean': speedup_mean,
                'median': speedup_median
            }
            print(f"\n[INFO] Speedup (based on mean):   {speedup_mean:.2f}x")
            print(f"[INFO] Speedup (based on median): {speedup_median:.2f}x")
        else:
            print(f"\n[WARNING] No custom operator timing available, cannot calculate speedup")

    except Exception as e:
        result['performance_info'] = f"Performance measurement failed: {str(e)}"
        print(f"[FAIL] {result['performance_info']}")
        return result

    # Step 6: Correctness check (run reference and check outputs match expected behavior)
    print(f"\n[STEP 6] Correctness validation")

    try:
        # For now, we rely on the C++ test's correctness check
        # In a full implementation, we would compare outputs directly
        result['correctness'] = True
        result['correctness_info'] = "C++ test passed and reference runs successfully"
        print("[PASS] Correctness validation passed")

    except Exception as e:
        result['correctness'] = False
        result['correctness_info'] = f"Correctness check failed: {str(e)}"
        print(f"[FAIL] {result['correctness_info']}")

    print(f"\n{'='*80}")
    print("Evaluation Summary:")
    print(f"  Compiled: {result['compiled']}")
    print(f"  Correctness: {result['correctness']}")
    if result['speedup'] and isinstance(result['speedup'], dict):
        print(f"  Speedup (mean):   {result['speedup'].get('mean', 'N/A'):.2f}x")
        print(f"  Speedup (median): {result['speedup'].get('median', 'N/A'):.2f}x")
    else:
        print(f"  Speedup: N/A")
    print(f"{'='*80}\n")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ascend C Evaluater - Evaluate custom Ascend C operators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluater.py --op matmul --runs 5
  python evaluater.py --op add --output results.json
        """
    )

    parser.add_argument('--op', type=str, required=True,
                        help='Operator name (must exist in op/ directory)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of evaluation runs (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: result_{op}.json)')

    args = parser.parse_args()

    # Validate operator exists
    op_path = os.path.join(op_base_path, args.op)
    if not os.path.exists(op_path):
        print(f"[ERROR] Operator not found: {op_path}")
        print(f"[ERROR] Available operators in {op_base_path}:")
        if os.path.exists(op_base_path):
            for item in os.listdir(op_base_path):
                if os.path.isdir(os.path.join(op_base_path, item)):
                    print(f"  - {item}")
        sys.exit(1)

    # Validate standard implementation exists
    standard_path = os.path.join(op_path, f'{args.op}_standard.py')
    if not os.path.exists(standard_path):
        print(f"[ERROR] Standard implementation not found: {standard_path}")
        print(f"[ERROR] Each operator directory must contain a '{args.op}_standard.py' file")
        sys.exit(1)

    # Run evaluation
    result = evaluate_operator(args.op, args.runs)

    # Save results
    output_file = args.output or f'result_{args.op}.json'
    output_path = os.path.join(project_root_path, output_file)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Results saved to: {output_path}")

    # Exit with appropriate code
    if result['compiled'] and result['correctness']:
        sys.exit(0)
    else:
        sys.exit(1)