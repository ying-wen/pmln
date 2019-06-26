# Pairwise Multi-Layer Nets for Learning Distributed Representation of Multi-field Categorical Data

This repo aims to provide an algorithm implementation for
Pairwise Multi-Layer Nets (PMLN) for Learning Distributed Representation of Multi-field Categorical Data and its baselines.

## Installation

1. Intsall the dependencies. 
 ```shell
sudo pip install keras, tensorflow
 ```
Note: The code was wrote in 2016, which means it requires (very old version) TensorFlow==0.10.0 and Keras==1.0.0.

2. Preprae data according to the instructions in data folder.

## Runing Experiments

```shell
python ctr_distcrimination.py % for FMLN-1
python ctr_distcrimination_concat.py % for FMLN-2
```
