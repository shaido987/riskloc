# RiskLoc
Code for the paper RiskLoc: Localization of Multi-dimensional Root Causes by Weighted Risk ([link](https://arxiv.org/abs/2205.10004)).  
Contains the implementation of RiskLoc and all baseline multi-dimensional root cause localization methods.

Implemented algorithms: RiskLoc, AutoRoot, [Squeeze](https://github.com/NetManAIOps/Squeeze), HotSpot, and Adtributor (normal and recursive).

## Requirements
- pandas
- numpy
- scipy
- kneed (for squeeze)
- loguru (for squeeze)

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
The supported algorithms are:
```
$ python run.py --help
usage: run.py [-h] {riskloc,autoroot,squeeze,old squeeze,hotspot,r_adtributor,adtributor} ...

RiskLoc

positional arguments: {riskloc,autoroot,squeeze,old squeeze,hotspot,r_adtributor,adtributor}

                        algorithm specific help
    riskloc             riskloc help
    autoroot            autoroot help
    squeeze             squeeze help
    hotspot             autoroot help
    r_adtributor        r_adtributor help
    adtributor          adtributor help

optional arguments:
  -h, --help            show this help message and exit
```
The code for Squeeze is adapted from the recently released code from the original publication: https://github.com/NetManAIOps/Squeeze.

To see the algorithm-specific arguments run: `python run.py 'algorithm' --help`. For example, for RiskLoc: 
```
$ python run.py riskloc --help
usage: run.py riskloc [-h] [--data-root DATA_ROOT] [--run-path RUN_PATH] [--derived [DERIVED]] [--n-threads N_THREADS] [--output-suffix OUTPUT_SUFFIX] [--debug [DEBUG]] [--risk-threshold RISK_THRESHOLD] [--ep-prop-threshold EP_PROP_THRESHOLD]

optional arguments:
  -h, --help                                  show this help message and exit
  --data-root DATA_ROOT                       root directory for all datasets (default ./data/)
  --run-path RUN_PATH                         directory or file to be run; 
                                              if a directory, any subdirectories will be considered as well;
                                	      must contain data-path as a prefix
  --derived [DERIVED]                         derived dataset (defaults to True for the D dataset and False for others)
  --n-threads N_THREADS                       number of threads to run
  --output-suffix OUTPUT_SUFFIX               suffix for output file
  --debug [DEBUG]                             debug mode
  --risk-threshold RISK_THRESHOLD             risk threshold
  --pep-threshold PEP_THRESHOLD               proportional explanatory power threshold
  --prune-elements [PRUNE_ELEMENTS]           use element pruning (True/False)
```

The `risk-threshold` and `pep-threshold` arguments are specific for the RiskLoc while the rest are shared by all algorithms. To see the algorithm-specific arguments for other algorithms simply run them with the `--help` flag or check the code in `run.py`.

## Datasets
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
```
@article{riskloc,
  title={RiskLoc: Localization of Multi-dimensional Root Causes by Weighted Risk},
  author={Kalander, Marcus},
  journal={arXiv preprint arXiv:2205.10004},
  year={2022}
}
```
