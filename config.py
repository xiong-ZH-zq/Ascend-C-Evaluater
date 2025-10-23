#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Configuration file for Ascend C Evaluater

import os

# Project paths
project_root_path = os.path.dirname(os.path.abspath(__file__))
op_base_path = os.path.join(project_root_path, 'op')

# Evaluation configuration
num_correct_trials = 5       # Number of correctness validation runs
num_perf_trials = 100        # Number of performance measurement runs
num_warmup = 3               # Number of warmup iterations

# Tolerance for correctness checking
atol = 1e-02                 # Absolute tolerance
rtol = 1e-02                 # Relative tolerance

# Performance measurement
seed_num = 1024              # Random seed for reproducibility

# Build configuration
temperature = 0.0            # Not used for now (for LLM generation compatibility)
top_p = 1.0                  # Not used for now (for LLM generation compatibility)

# Device configuration
device_type = 'npu'          # Device type: 'npu' for Ascend NPU
device_id = 0                # Device ID to use
