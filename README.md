# RiskLoc
This repository contains code for the paper [RiskLoc: Localization of Multi-dimensional Root Causes by Weighted Risk](https://arxiv.org/abs/2205.10004). Both the implementation of RiskLoc itself and all baseline multi-dimensional root cause localization methods in the paper are included, as well as the code to generate synthetic datasets as described in the paper.

<p align="center">
  <img width="736" alt="architecture" src="https://github.com/shaido987/riskloc/assets/1130029/c9b8d791-ac94-4edc-b70f-8555467b6c2a">
</p>

**Short problem description:**  
RiskLoc solves the problem of identifying the root cause of an anomaly occurring in a time series with multi-dimensional attributes. These types of time series can be regarded as aggregations (the total sum in the simplest case) of numerous underlying, more fine-grained, time series.   

For example, a time series T with 2 dimensions (d1 and d2), each with 3 possible values: 
- d1: [a, b, c]
- d2: [d, e, f]

is built up of 9 fine-grained time series (two examples of these are the time series corresponding to {d1: a, d2: d} and {d1: b, d2: f}). 

The goal is to find the specific dimension and dimensional values (the elements) of the root cause when an error occurs in the fully aggregated time series T. This is a search problem where any combination of dimensions and values are considered, and there can be multiple elements in the fanal root cause set. For the example time series above, one potential root cause set can be {{d1: a, d2: [d, e]}, {d1: b, d2: e}}. Since any combination and any number of elements needs to be considered, the total search space is huge which is the main challenge.

## Requirements
- pandas
- numpy
- scipy
- kneed (for squeeze)
- loguru (for squeeze)
- pyyaml

## How to run

To run, use the `run.py` file. There are a couple of options, either to use a single file or to run all files in a directory (including all subdirectories).

Example of running a single file using RiskLoc in debug mode:
```
python run.py riskloc --run-path /data/B0/B_cuboid_layer_1_n_ele_1/1450653900.csv --debug
```

Example of running all files in a particular setting for a dataset (setting derived to True):
```
python run.py riskloc --run-path /data/D/B_cuboid_layer_3_n_ele_3 --derived
```

Example of running all files in a dataset:
```
python run.py riskloc --run-path /data/B0
```

Example of running all datasets with 20 threads:
```
python run.py riskloc --n-threads 20
```

Changing `riskloc` to any of the supported algorithms will run those instead, see below.

## Algorithms 
Implemented algorithms: RiskLoc, AutoRoot, RobustSpot, Squeeze, HotSpot, and Adtributor (normal and recursive).

They can be run by specifying the algorithm name as the first input parameter to the `run.py` file:
```
$ python run.py --help
usage: run.py [-h] {riskloc,autoroot,squeeze,old squeeze,hotspot,r_adtributor,adtributor} ...

RiskLoc

positional arguments: {riskloc,autoroot,robustspot,squeeze,hotspot,r_adtributor,adtributor}

                        algorithm specific help
    riskloc             riskloc help
    autoroot            autoroot help
    robustspot          robustspot help
    squeeze             squeeze help
    hotspot             autoroot help
    r_adtributor        r_adtributor help
    adtributor          adtributor help

optional arguments:
  -h, --help            show this help message and exit
```
The code for Squeeze is adapted from the released code from the original publication: https://github.com/NetManAIOps/Squeeze.
The code for RobustSpot is similarly adapted from their recently released code: https://github.com/robustspotproject/RobustSpot.

To see the algorithm-specific arguments run: `python run.py 'algorithm' --help`. For example, for RiskLoc: 
```
$ python run.py riskloc --help
usage: run.py riskloc [-h] [--data-root DATA_ROOT] [--run-path RUN_PATH] [--derived [DERIVED]] [--n-threads N_THREADS] [--output-suffix OUTPUT_SUFFIX] [--debug [DEBUG]] [--risk-threshold RISK_THRESHOLD] [--pep-threshold PEP_THRESHOLD] [--n-remove N_REMOVE] [--remove-relative [REMOVE_RELATIVE]] [--prune-elements [PRUNE_ELEMENTS]]

options:
  -h, --help                           show this help message and exit
  --data-root DATA_ROOT                root directory for all datasets (default ./data)
  --run-path RUN_PATH                  directory or file to be run; if a directory, any subdirectories will be considered as well;
                                       must contain data-path as a prefix
  --derived [DERIVED]                  derived dataset (defaults to True for the D and RS datasets and False for others)
  --n-threads N_THREADS                number of threads to run
  --output-suffix OUTPUT_SUFFIX        suffix for output csv file
  --debug [DEBUG]                      debug mode
  --risk-threshold RISK_THRESHOLD      risk threshold
  --pep-threshold PEP_THRESHOLD        proportional explanatory power threshold
  --n-remove N_REMOVE                  number of elements to ignore when computing the cutoff point
  --remove-relative [REMOVE_RELATIVE]  if true then n_remove is a percentage value
  --prune-elements [PRUNE_ELEMENTS]    use element pruning (True/False)
```

The `risk-threshold` and below arguments are specific for the RiskLoc while the rest are shared by all algorithms. To see the algorithm-specific arguments for other algorithms simply run them with the `--help` flag or check the code in `run.py`.

## Datasets
The real-world dataset with derived measures from RobustSpot (RS) is already present in the data folder and can be used immediately.

The semi-synthetic datasets can be downloaded from: https://github.com/NetManAIOps/Squeeze.
To run these, place them within the data/ directory and name them: A, B0, B1, B2, B3, B4, and D, respectively.

The three synthetic datasets used in the paper can be generated using `generate_dataset.py` as follows.

S dataset:
```
python generate_dataset.py --num 1000 --dataset-name S --seed 121
```
L dataset:
```
python generate_dataset.py --num 1000 --dataset-name L --seed 122 --dims 10 24 10 15 --noise-level 0.0 0.1 --anomaly-severity 0.5 1.0 --anomaly-deviation 0.0 0.0 --num-anomaly 1 5 --num-anomaly-elements 1 1 --only-last-layer
```
H dataset:
```
python generate_dataset.py --num 100 --dataset-name H --seed 123 --dims 10 5 250 20 8 12
```

In addition, new, interesting datasets can be created using `generate_dataset.py` for extended empirical verification and research purposes. Supported input arguments can be found at the beginning of the `generate_dataset.py` file or using the `--help` flag. 

## Citation
If you find this code useful, please cite the following paper:

```
@article{riskloc,
  title={RiskLoc: Localization of Multi-dimensional Root Causes by Weighted Risk},
  author={Kalander, Marcus},
  journal={arXiv preprint arXiv:2205.10004},
  year={2022}
}
```
