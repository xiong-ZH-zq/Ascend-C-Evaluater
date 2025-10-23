# Ascend C Evaluater

## Introduction

This project is a simple evaluater which is similar to evaluation part in [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench).

The structure of this project:

```bash
├── config.py
├── dataset.py
├── evaluater.py      # main script
├── IMPLEMENTATION_SUMMARY.md
├── op                # Operator projects that needs to be tested
│   ├── add
│   │   ├── add_custom.cpp
│   │   ├── add_custom_test.py
│   │   ├── CMakeLists.txt
│   │   ├── pybind11.cpp
│   │   ├── README.md
│   │   └── run.sh
│   └── matmul
│       ├── CMakeLists.txt
│       ├── matmul_custom.cpp
│       ├── matmul_custom_test.py
│       ├── matmul_custom_tiling.cpp
│       ├── pybind11.cpp
│       ├── README.md
│       └── run.sh
├── README.md
├── reference           # Standard torch operator
│   ├── add.py
│   └── matmul.py
└── utils.py
```


## Usage

### Command

You can run a simple example by command below:

```shell
python evaluater.py --op matmul --reference matmul --runs 5
```

- `--op`: Name of the task.
- `--reference`: Reference task name.
- `--runs`: How many turns should be run to get performance evaluation.

### Workflow

To test the correctness and speed of LLM-generated operator, you may follow the workflow below (Take `matmul` for example):

1. Given `op/matmul/CMakeLists.txt`, `op/matmul/pybind11.cpp`, `op/matmul/run.sh` and `op/matmul/matmul_custom_test.py`, let LLM generate `.txt` project file that contains the following parts:
    - `matmul_custom_tiling.cpp`
    - `matmul_custom.cpp`
2. Replace them and run command below:
```shell
python evaluater.py --op matmul --reference matmul
```
3. Return result file: `result_matmul.json` to LLM and run this workflow again.

> **WARNING**: 
> You may need to delete existed `op/matmul/build` directory before starting this workflow to avoid `CMakeError`.

## TODO

- [ ] Data Input Customization.
- [ ] Reference ❌
- [ ] Instruction (Workflow).
- [ ] Preparation Instruction (dataset, op).