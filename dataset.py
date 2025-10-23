#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operator Registry (Optional - Not Currently Used)

This file is OPTIONAL and NOT required for the evaluator to work.
It is kept for potential future use cases such as:
- Batch evaluation of multiple operators
- Operator categorization and grouping
- Documentation and discovery tools
- Automated testing across operator categories

IMPORTANT: You do NOT need to update this file when adding new operators.
The evaluator automatically discovers operators by checking the op/ directory.

To add a new operator:
1. Create op/{operator_name}/ directory
2. Add {operator_name}_input_config.py
3. Add {operator_name}_standard.py
4. Add {operator_name}_custom_test.py
5. Run: python evaluater.py --op {operator_name}

That's it! No need to modify this file.
"""

# Define available operators for evaluation (Optional metadata)
dataset = {
    "add": {
        "category": "elementwise",
        "description": "Element-wise addition operation"
    },
    "matmul": {
        "category": "linear_algebra",
        "description": "Matrix multiplication with bias operation"
    },
    # Add more operators here if you want to maintain this registry
    # But again, this is OPTIONAL - the evaluator doesn't use it
}

# Category to example operation mapping (Optional)
category2exampleop = {
    "elementwise": "add",
    "linear_algebra": "matmul",
}

# Auto-discover available operators from filesystem (Example usage)
def get_available_operators():
    """
    Automatically discover available operators by scanning op/ directory.
    This is how the evaluator actually works.
    """
    import os
    op_dir = os.path.join(os.path.dirname(__file__), 'op')
    if not os.path.exists(op_dir):
        return []

    operators = []
    for item in os.listdir(op_dir):
        item_path = os.path.join(op_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            # Check if it has the required files
            has_standard = os.path.exists(os.path.join(item_path, f'{item}_standard.py'))
            has_test = os.path.exists(os.path.join(item_path, f'{item}_custom_test.py'))
            if has_standard and has_test:
                operators.append(item)

    return sorted(operators)


if __name__ == "__main__":
    # Example: List all available operators
    print("Available operators:")
    for op in get_available_operators():
        info = dataset.get(op, {"category": "unknown", "description": "No description"})
        print(f"  - {op:15s} [{info['category']:20s}] {info['description']}")
