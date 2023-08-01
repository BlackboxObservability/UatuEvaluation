# Blackbox Observability of Features and Feature Interactions

This repository is the supplementary Web site for the paper "Blackbox Observability of Features and Feature Interactions" submitted to the ICSE. In this repository, we list further information to the paper. 
## Requirements
To replicate our results, one has to install the following python packages:
* [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [dd](https://github.com/tulip-control/dd) 
    
    **Note:** To be able to reproduce our results you have to install the cython extension CUDD of this package
* [argparse](https://pypi.org/project/argparse/)


## Data

The data to all subject systems used to answer RQ_1 and RQ_2 are included in the directory [examples](./examples). In each of the case study directories, you will find the following files:
* *README.md*: This file includes information about the performed experiments and the configurable systems we used.
* *utils.py*: This file contains some useful functions that are used in the evaluation_runner
* *parser.py*: This file contains the parser for the experiment feature models (dnf files)
* *observability.py*: This file contains the implementation of our notion of blackbox observability.
* *evaluation_runner.py*: The driver for the evaluation, you can run single experiments, or all experiments for RQ1 and RQ2, to run the experiments for RQ3 you also have to clone the repository [Performance Evolution Website](https://github.com/ChristianKaltenecker/PerformanceEvolution_Website) to the same directory as this repository.


We used the given data and the scripts in this directory to generate all results.


## RQ3: What fraction of features and feature interactions in state-of-the art blackbox performance models is observable?

The case studies for the experiments (listed below) can be found in the gitlab repository on the website [Performance Evolution Website](https://github.com/ChristianKaltenecker/PerformanceEvolution_Website).

Case Studies | 
---
brotli|
Fast Downward|
HSQLDB|
lrzip|
MariaDB |
MySQL |
OpenVPN|
Opus|
PostgreSQL|
VP8|
z3|
