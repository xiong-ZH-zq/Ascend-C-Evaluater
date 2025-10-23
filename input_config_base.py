#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Configuration Base Module

This module provides a base class for defining operator input configurations.
Separates input data generation from operator implementation.

Usage:
    class MatmulInputConfig(BaseInputConfig):
        def generate_inputs(self):
            M, K, N = 1024, 256, 640
            A = torch.randn(M, K, dtype=torch.float16)
            B = torch.randn(K, N, dtype=torch.float16)
            bias = torch.randn(N, dtype=torch.float32)
            return [A, B, bias]
"""

import torch


class BaseInputConfig:
    """
    Base class for operator input configuration.

    Features:
    - Define input shapes and dtypes in one place
    - Reusable across correctness test and performance test
    - Support for random seed control
    - Flexible device placement
    """

    def __init__(self, seed=None):
        """
        Initialize input configuration.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        try:
            import torch_npu
            if torch.npu.is_available():
                torch.npu.manual_seed(seed)
        except ImportError:
            pass

    def generate_inputs(self):
        """
        Generate input tensors for the operator.

        Override this method to define your operator's inputs.

        Returns:
            list: List of input tensors (CPU)

        Example:
            def generate_inputs(self):
                x = torch.randn(1024, 1024, dtype=torch.float16)
                y = torch.randn(1024, 1024, dtype=torch.float16)
                return [x, y]
        """
        raise NotImplementedError("Must implement generate_inputs()")

    def to_device(self, tensors, device):
        """
        Move tensors to specified device.

        Args:
            tensors: List of tensors
            device: Target device (torch.device or string)

        Returns:
            list: Tensors on target device
        """
        if isinstance(device, str):
            device = torch.device(device)

        result = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                result.append(tensor.to(device))
            else:
                result.append(tensor)
        return result

    def get_inputs_for_device(self, device='cpu'):
        """
        Generate inputs and move to device.

        Args:
            device: Target device ('cpu', 'cuda:0', 'npu:0', etc.)

        Returns:
            list: Input tensors on target device
        """
        inputs = self.generate_inputs()
        return self.to_device(inputs, device)

    def get_input_shapes(self):
        """
        Get shapes of all inputs (without generating actual tensors).

        Override this for documentation/validation purposes.

        Returns:
            list: List of tuples representing shapes

        Example:
            def get_input_shapes(self):
                return [(1024, 256), (256, 640), (640,)]
        """
        inputs = self.generate_inputs()
        return [tuple(inp.shape) if isinstance(inp, torch.Tensor) else None for inp in inputs]

    def get_input_dtypes(self):
        """
        Get dtypes of all inputs.

        Returns:
            list: List of dtypes
        """
        inputs = self.generate_inputs()
        return [inp.dtype if isinstance(inp, torch.Tensor) else type(inp) for inp in inputs]

    def __repr__(self):
        """String representation of input configuration"""
        shapes = self.get_input_shapes()
        dtypes = self.get_input_dtypes()
        lines = ["Input Configuration:"]
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            lines.append(f"  Input {i}: shape={shape}, dtype={dtype}")
        return "\n".join(lines)
