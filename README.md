# Out of Distribution Generalization for Problems in Mechanics

This repository contains the code for the following paper:

> Lingxiao Yuan, Emma Lejuene\*, Harold Park\*
>
> [Out of Distribution Generalization for Problems in Mechanics](link forthcoming)

## Datasets
The experiments use the following dataset. The datasets contains six benchmark out of distirbution datasets specifically for regression problems in mechanics.  
- [Mechanical MNIST – Distribution Shift](https://open.bu.edu/handle/2144/44485)


## Details 
`model`
This folder contains the information of reproducing the results from the paper. 
* `model.py` : The two Machine Learning models(MLP and modified LeNet) for implementing all algorithms
* `make_envs.py` : The file loads dataset and makes a list containing data from different environments
* `main.py` : The whole framework of training process using different algorithms for out of distribution problems
* `evaluation.py` : demo code for evaluating the trained model on covariate shift dataset
* `submit_job.sh` : submit job for training models by selected hyperparameters 

`envs_details`
This folder contains more details of the process of how we generate or select OOD dataset. This information is not needed for reproducing the results from the paper. 

* `sampling_bias_demo.ipynb` : The demo of creating data distrbution dataset caused by sampling bias 
* `make_envs_sampling_bias_MNIST.py` : The details of generating sampling bias dataset from Mechanical MNIST dataset
* `make_envs_sampling_bias_EMNIST.py` : The details of generating sampling bias dataset from Mechanical MNIST - EMNIST Letters dataset
* `index_dataset_MNIST_EMNIST.txt` : the corresponding index of the OOD data in the original MNIST and EMNIST Letters

### Hypermeters 
Below are the parameters to choose for training the model. The selected hyperparamters were given in the paper. But the results are not guaranteed to be exactly the same due to randomness. 

* `lr`: learning rate
* `steps`: the whole epochs for training the model
* `penalty_anneal_iters`: the anneal step after which we add the penalty to the loss function
* `penalty_weight`: the weight of penalty term, this controls the strength of the penalty term
* `algorithm`: the name of the algorithm to implemente (`erm`, `irm`,`rex`,`iga`)
* `model`: the name of the model to train(`mlp`,`lenet`)