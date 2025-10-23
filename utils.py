#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Utility functions for Ascend C Evaluater

import os
import subprocess
import time
import numpy as np
import torch
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    print("[WARNING] torch_npu not available, NPU operations will not work")


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if NPU_AVAILABLE and torch.npu.is_available():
        torch.npu.manual_seed(seed)


def build_operator(op_path):
    """
    Build an Ascend C operator using CMake build system.

    Args:
        op_path: Path to the operator directory

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    if not os.path.exists(op_path):
        return False, f"Operator path does not exist: {op_path}"

    cmakelists = os.path.join(op_path, 'CMakeLists.txt')
    if not os.path.exists(cmakelists):
        return False, f"CMakeLists.txt not found: {cmakelists}"

    print(f"[INFO] Building operator at {op_path}")
    original_dir = os.getcwd()

    try:
        # Create and enter build directory
        build_dir = os.path.join(op_path, 'build')
        os.makedirs(build_dir, exist_ok=True)
        os.chdir(build_dir)

        # Remove ASCEND_CUSTOM_OPP_PATH to avoid build interference
        env = os.environ.copy()
        env.pop('ASCEND_CUSTOM_OPP_PATH', None)

        # Run cmake
        cmake_result = subprocess.run(
            ['cmake', '..'],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        if cmake_result.returncode != 0:
            return False, f"CMake failed:\n{cmake_result.stderr}"

        # Run make
        make_result = subprocess.run(
            ['make'],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        if make_result.returncode != 0:
            error_output = ''
            for line in make_result.stdout.split('\n') + make_result.stderr.split('\n'):
                if '[ERROR]' in line or 'error:' in line.lower():
                    error_output += line + '\n'
            return False, f"Make failed:\n{error_output if error_output else make_result.stderr}"

        print("[INFO] Build succeeded")
        return True, None

    except subprocess.TimeoutExpired:
        return False, "Build timed out after 5 minutes"
    except Exception as e:
        return False, f"Build error: {str(e)}"
    finally:
        os.chdir(original_dir)


def run_cpp_test(op_path, num_trials=100):
    """
    Run the Python test for an operator (using pybind11 module).

    Args:
        op_path: Path to the operator directory
        num_trials: Number of performance measurement trials

    Returns:
        tuple: (success: bool, results: dict or error_message: str)
    """
    import json

    build_dir = os.path.join(op_path, 'build')

    if not os.path.exists(build_dir):
        return False, "Build directory not found"

    # Find the test Python file
    test_files = [f for f in os.listdir(op_path) if f.endswith('_test.py')]
    if not test_files:
        return False, "Test file not found (expected *_test.py)"

    test_file = os.path.join(op_path, test_files[0])
    print(f"[INFO] Running Python test: {test_file}")
    print(f"[INFO] Performance trials: {num_trials}")

    original_dir = os.getcwd()

    try:
        os.chdir(build_dir)

        # Remove old timing results if they exist
        timing_file = os.path.join(build_dir, 'timing_results.json')
        if os.path.exists(timing_file):
            os.remove(timing_file)

        # Set environment variable for num_trials
        env = os.environ.copy()
        env['PERF_NUM_TRIALS'] = str(num_trials)

        # Run the test file
        result = subprocess.run(
            ['python3', test_file],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        # Parse output for correctness
        output = result.stdout + result.stderr

        # Check test results
        correctness = None
        if result.returncode == 0:
            # Check for test pass indicators
            if 'OK' in output or 'PASSED' in output or 'Ran 1 test' in output:
                correctness = True
        else:
            correctness = False

        results = {
            'correctness': correctness,
            'num_trials': num_trials,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

        # Try to read timing results from JSON file
        if os.path.exists(timing_file):
            try:
                with open(timing_file, 'r') as f:
                    timing_data = json.load(f)
                    results.update(timing_data)
                    stats = timing_data.get('statistics', {})
                    if stats:
                        print(f"[INFO] Custom operator timing:")
                        print(f"       Mean: {stats.get('mean', 0):.3f} ms")
                        print(f"       Median: {stats.get('median', 0):.3f} ms")
            except Exception as e:
                print(f"[WARNING] Failed to load timing results: {str(e)}")

        return True, results

    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Test execution error: {str(e)}"
    finally:
        os.chdir(original_dir)


def measure_pytorch_performance(model, inputs, device, num_warmup=3, num_trials=100):
    """
    Measure PyTorch model performance on NPU.

    Args:
        model: PyTorch model
        inputs: List of input tensors
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement iterations

    Returns:
        dict: Performance metrics with raw_data and statistics
    """
    import time

    model = model.to(device)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*inputs)
            if NPU_AVAILABLE and device.type == 'npu':
                torch_npu.npu.synchronize(device=device)
            elif device.type == 'cuda':
                torch.cuda.synchronize(device=device)

    # Measurement
    elapsed_times = []

    if NPU_AVAILABLE and device.type == 'npu':
        for i in range(num_trials):
            # Clear cache before each measurement
            torch_npu.npu.empty_cache()
            torch_npu.npu.synchronize(device=device)

            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)

            with torch.no_grad():
                start_event.record()
                _ = model(*inputs)
                end_event.record()

            torch_npu.npu.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == num_trials:
                print(f"[PERF] Standard PyTorch: Completed {i + 1}/{num_trials} trials")
    else:
        # Fallback to CPU timing
        for i in range(num_trials):
            # Clear cache if CUDA is available
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device=device)

            with torch.no_grad():
                start_time = time.time()
                _ = model(*inputs)
                end_time = time.time()
                elapsed_times.append((end_time - start_time) * 1000)  # Convert to ms

            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == num_trials:
                print(f"[PERF] Standard implementation: Completed {i + 1}/{num_trials} trials")

    # Calculate statistics
    elapsed_times_sorted = sorted(elapsed_times)
    mean_val = float(np.mean(elapsed_times))

    # Calculate median
    n = len(elapsed_times_sorted)
    if n % 2 == 0:
        median_val = (elapsed_times_sorted[n//2 - 1] + elapsed_times_sorted[n//2]) / 2
    else:
        median_val = elapsed_times_sorted[n//2]

    return {
        'raw_data': elapsed_times,
        'statistics': {
            'mean': mean_val,
            'median': float(median_val),
            'std': float(np.std(elapsed_times)),
            'min': float(np.min(elapsed_times)),
            'max': float(np.max(elapsed_times)),
            'num_trials': num_trials
        }
    }


def check_correctness(reference_output, custom_output, atol=1e-02, rtol=1e-02):
    """
    Check if custom output matches reference output within tolerance.

    Args:
        reference_output: Reference output tensor
        custom_output: Custom implementation output tensor
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        tuple: (correctness: bool, info: str)
    """
    if not isinstance(reference_output, torch.Tensor) or not isinstance(custom_output, torch.Tensor):
        return False, "Outputs must be torch tensors"

    if reference_output.shape != custom_output.shape:
        return False, f"Shape mismatch: reference {reference_output.shape} vs custom {custom_output.shape}"

    try:
        if torch.allclose(reference_output, custom_output, atol=atol, rtol=rtol):
            max_diff = torch.max(torch.abs(reference_output - custom_output)).item()
            return True, f"Outputs match within tolerance (max diff: {max_diff:.6f})"
        else:
            max_diff = torch.max(torch.abs(reference_output - custom_output)).item()
            mean_diff = torch.mean(torch.abs(reference_output - custom_output)).item()
            return False, f"Output mismatch (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})"
    except Exception as e:
        return False, f"Comparison error: {str(e)}"
