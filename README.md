# Out of Distribution for Problems in Mechanics

This repository contains the code for the following paper:

> Lingxiao Yuan, Emma Lejuene\*, Harold Park\*
>
> [Out of Distribution Generalization for Problems in Mechanics](link forthcoming)


The experiments use the following dataset. The datasets contains six benchmark out of distirbution datasets specifically for regression problems in mechanics.  
- [Mechanical MNIST – Distribution Shift](https://open.bu.edu/handle/2144/44485)


## Details 
`generate_data_list`

* `model.py` : The two Machine Learning models(MLP and modified LeNet) for implementing all algorithms
* `make_envs.py` : The file loads dataset and makes a list containing data from different environments
* `main.py` : The whole framework of training process using different algorithms for out of distribution problems
* `sampling_bias_demo.ipynb` : The demo of creating data distrbution dataset caused by sampling bias
* `evaluation.py` : code for evaluating the trained model on test environments

## Training


## Testing