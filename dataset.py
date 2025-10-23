#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Dataset definition for Ascend C Evaluater

# Define available operators for evaluation
dataset = {
    "add": {
      "category": "add",
      "description": "Matrix add operation"  
    },
    "matmul": {
        "category": "matmul",
        "description": "Matrix multiplication operation"
    },
    # Add more operators here as they become available
}

# Category to example operation mapping
category2exampleop = {
    "add": "add",
    "matmul": "matmul",
}
