#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matmul Operator Input Configuration

Defines input shapes, dtypes, and generation logic for matmul operator.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from input_config_base import BaseInputConfig


class MatmulInputConfig(BaseInputConfig):
    """Input configuration for matmul operator: C = A * B + bias"""

    def __init__(self, M=1024, K=256, N=640, seed=None):
        """
        Initialize matmul input configuration.

        Args:
            M: Number of rows in A
            K: Number of columns in A, rows in B
            N: Number of columns in B
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.M = M
        self.K = K
        self.N = N

    def generate_inputs(self):
        """
        Generate inputs for matmul: [A, B, bias]

        Returns:
            list: [A(M,K), B(K,N), bias(N,)]
        """
        A = torch.randn(self.M, self.K, dtype=torch.float16)
        B = torch.randn(self.K, self.N, dtype=torch.float16)
        bias = torch.randn(self.N, dtype=torch.float32)
        return [A, B, bias]

    def get_input_shapes(self):
        """Return input shapes without generating tensors"""
        return [(self.M, self.K), (self.K, self.N), (self.N,)]
