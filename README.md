# A differentiable program synthesizer for SyGuS

## Introduction

We propose an end-to-end differentiable program synthesizer for SyGuS. More details of the algorithm are given in [Differentiable Synthesis of Program Architectures](https://proceedings.neurips.cc/paper/2021/file/5c5a93a042235058b1ef7b0ac1e11b67-Paper.pdf) with its implementation available at [dPads](https://github.com/RU-Automated-Reasoning-Group/dPads).


## Requirements
- Python 3.6+
- PyTorch 1.4.0+
- scikit-learn 0.22.1+
- Numpy
- tqdm


## Setup

```
cd metal/spec_encoder/
make
```

```
export PYTHONPATH=your_path_to/DifferentiableSyGuS/
```

## Run (-single_sample to specify a benchmark) 

The tool currently only supports the CrCi set (cryptography circuits) benchmarks. As the synthesizer is differentiable end-to-end, we require smooth program semantics. There are two ways to approximate the circuit semantics (for and, or, xor, and not), defined in library_functions.py. 

Each of the following two commands invokes one of the two semantics definitions. There are benchmarks that can only be solved by one of the semantics definitions. Therefore, to evaluate the full strength of the synthesizer, **both** commands should be applied to each benchmark (log.txt provides the result of applying EUSover as a baseline).

```
python3 archsyn/new_main.py --batch_size 50 --random_seed 0 -data_root ./benchmarks -file_list all -single_sample CrCy_10-sbox2-D5-sIn30.sl -top_left True -GM True --sem minmax
```

```
python3 archsyn/new_main.py --batch_size 50 --random_seed 0 -data_root ./benchmarks -file_list all -single_sample CrCy_10-sbox2-D5-sIn30.sl -top_left True -GM True --sem arith
```
