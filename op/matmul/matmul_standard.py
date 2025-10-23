#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard PyTorch Implementation for Matmul Operator

This module defines ONLY the operator logic.
Input configuration is in matmul_input_config.py
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard PyTorch matmul operator: C = A * B + bias

    This is used for:
    1. Correctness verification (in custom test)
    2. Performance baseline (in evaluater)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication with bias.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)
            bias: Bias tensor of shape (N,)

        Returns:
            Output tensor of shape (M, N)
        """
        return torch.matmul(A, B) + bias


# For backward compatibility with evaluater.py
def get_init_inputs():
    """No initialization inputs needed"""
    return []


def get_inputs():
    """
    Legacy function for backward compatibility.
    New code should use matmul_input_config.py instead.
    """
    import sys
    import os
    # Add op directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    from matmul_input_config import MatmulInputConfig
    config = MatmulInputConfig()
    return config.generate_inputs()
