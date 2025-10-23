#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard PyTorch Implementation for Add Operator

This module defines ONLY the operator logic.
Input configuration is in add_input_config.py
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Standard PyTorch add operator: C = A + B

    This is used for:
    1. Correctness verification (in custom test)
    2. Performance baseline (in evaluater)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs element-wise addition.

        Args:
            A: Input tensor
            B: Input tensor (same shape as A)

        Returns:
            Output tensor (same shape as A)
        """
        return torch.add(A, B)


# For backward compatibility with evaluater.py
def get_init_inputs():
    """No initialization inputs needed"""
    return []


def get_inputs():
    """
    Legacy function for backward compatibility.
    New code should use add_input_config.py instead.
    """
    import sys
    import os
    # Add op directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    from add_input_config import AddInputConfig
    config = AddInputConfig()
    return config.generate_inputs()
