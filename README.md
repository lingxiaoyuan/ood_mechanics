# Out of Distribution Generalization for Problems in Mechanics

This repository contains the code for the following paper:

> Lingxiao Yuan, Harold Park\*, Emma Lejeune\*
>
> [Out of Distribution Generalization for Problems in Mechanics](https://arxiv.org/abs/2206.14917)

## Datasets
The experiments use the following dataset. The datasets contains six benchmark out of distirbution datasets specifically for regression problems in mechanics.  
- [Mechanical MNIST – Distribution Shift](https://open.bu.edu/handle/2144/44485)

<img src="https://user-images.githubusercontent.com/58915487/171908617-18cdc387-8b24-4cbb-869b-d0538379de01.png" alt="Mechanical MNIST – Distribution Shift" width="400"/>

## Details 
`envs_details`
This folder contains more details of the process of how we generate and select data for OOD dataset. This information is not needed for reproducing the results from the paper. 

* `sampling_bias_demo.ipynb` : The demo of creating data distrbution dataset caused by sampling bias 
* `make_envs_sampling_bias_MNIST.py` : The demo code for generating sampling bias dataset from Mechanical MNIST dataset
* `index_dataset_MNIST_EMNIST.txt` : The corresponding index of the OOD data in the original MNIST and EMNIST Letters

`model`
This folder contains the information of reproducing the results from the paper. 
* `model.py` : The two Machine Learning models (MLP and modified LeNet) for implementing all algorithms
* `make_envs.py` : The file loads dataset and makes a list containing data from different environments
* `main.py` : The whole framework of training process using different algorithms for out of distribution problems
* `evaluation.py` : Demo code for evaluating the trained model on covariate shift dataset
* `submit_job.sh` : Submit job for training models by selected hyperparameters 



## Hypermeters 
Below are the parameters to choose for training the model. The selected hyperparamters were given in the paper. But the results are not guaranteed to be exactly the same due to randomness. 

* `lr`: learning rate
* `steps`: the whole epochs for training the model
* `penalty_anneal_iters`: the anneal step after which we add the penalty to the loss function
* `penalty_weight`: the weight of penalty term, this controls the strength of the penalty term
* `algorithm`: the name of the algorithm to implemente (`erm`, `irm`,`rex`,`iga`)
* `model`: the name of the model to train(`mlp`,`lenet`)

## References
The `main.py` file adapted three algorithms designed for out of distribution problems. Below are the links to these paper and the source code for the IRM algorithms that we adapted. 
* IRM:[Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)

  Code for IRM:https://github.com/facebookresearch/InvariantRiskMinimization, licensed under [LICENCE](https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/LICENSE)
* REx:[Out-of-Distribution Generalization via Risk Extrapolation (REx)](https://arxiv.org/abs/2003.00688)
* IGA:[Out-of-Distribution Generalization with Maximal Invariant Predictor](https://openreview.net/forum?id=FzGiUKN4aBp&referrer=[the%20profile%20of%20Masanori%20Koyama](/profile?id=~Masanori_Koyama1))

## Links
Code for implementing finite element simulation of heterogeneous material subject to large deformation due to equibiaxial extension is available on GitHub, the environmental control parameter $s$ can be adjusted by changing the variable 'high' (line 62) to other value of $s$ (orginally $100$). More details of parameter $s$ can be found in the [paper](link forthcoming). The link to the github code is: https://github.com/elejeune11/Mechanical-MNIST/blob/master/generate_dataset/Equibiaxial_Extension_FEA_test_FEniCS.py

The Mechanical MNIST - Equibiaxial Extension Collection from which sampling bias data is created is available on OpenBU Institutional Repository at: https://open.bu.edu/handle/2144/39428


