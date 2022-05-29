import numpy as np
import torch

def env_covariate_shift():
    
    E100_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s100_input.txt')
    E100_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s100_label.txt')
    
    E90_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s90_input.txt')
    E90_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s90_label.txt')
    
    E75_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s75_input.txt')
    E75_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s75_label.txt')

    E50_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s50_input.txt')
    E50_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s50_label.txt')

    envs = []

    envs.append({'images':torch.tensor(E100_input,dtype = torch.float32), 
                 'labels':torch.tensor(E100_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E90_input,dtype = torch.float32),
                 'labels':torch.tensor(E90_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E75_input,dtype = torch.float32), 
                 'labels':torch.tensor(E75_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E50_input,dtype = torch.float32), 
                 'labels':torch.tensor(E50_label,dtype = torch.float32)})

    return envs

def env_mechanism_shift():
    
    E100_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s100_input.txt')
    E100_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s100_label.txt')
    
    E90_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s90_input.txt')
    E90_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s90_label.txt')
    
    E25_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s25_input.txt')
    E25_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s25_label.txt')
    
    E10_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s10_input.txt')
    E10_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s10_label.txt')

    envs = []

    envs.append({'images':torch.tensor(E100_input,dtype = torch.float32), 
                 'labels':torch.tensor(E100_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E90_input,dtype = torch.float32),
                 'labels':torch.tensor(E90_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E25_input,dtype = torch.float32), 
                 'labels':torch.tensor(E25_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(E10_input,dtype = torch.float32), 
                 'labels':torch.tensor(E10_label,dtype = torch.float32)})

    return envs

def env_sampling_bias():
    
    r15_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r15_input.txt')
    r15_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r15_label.txt')
    
    r2neg_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r2neg_input.txt')
    r2neg_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r2neg_label.txt')
    
    r5neg_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r5neg_input.txt')
    r5neg_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r5neg_label.txt')
    
    r10neg_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r10neg_input.txt')
    r10neg_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r10neg_label.txt')
    
    r1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r1_input.txt')
    r1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r1_label.txt')

    envs = []

    envs.append({'images':torch.tensor(r15_input,dtype = torch.float32), 
                 'labels':torch.tensor(r15_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(r2neg_input,dtype = torch.float32),
                 'labels':torch.tensor(r2neg_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(r5neg_input,dtype = torch.float32),
                 'labels':torch.tensor(r5neg_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(r10neg_input,dtype = torch.float32),
                 'labels':torch.tensor(r10neg_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(r1_input,dtype = torch.float32),
                 'labels':torch.tensor(r1_label,dtype = torch.float32)})

    return envs


