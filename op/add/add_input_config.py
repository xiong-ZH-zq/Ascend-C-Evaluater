#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Operator Input Configuration

Defines input shapes, dtypes, and generation logic for add operator.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from input_config_base import BaseInputConfig


class AddInputConfig(BaseInputConfig):
    """Input configuration for add operator: C = A + B"""

    def __init__(self, shape=(8, 4096), seed=None):
        """
        Initialize add input configuration.

        Args:
            shape: Shape of input tensors
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.shape = shape

    def generate_inputs(self):
        """
        Generate inputs for add: [A, B]

        Returns:
            list: [A(shape), B(shape)]
        """
        A = torch.randn(*self.shape, dtype=torch.float16)
        B = torch.randn(*self.shape, dtype=torch.float16)
        return [A, B]

    def get_input_shapes(self):
        """Return input shapes without generating tensors"""
        return [self.shape, self.shape]
