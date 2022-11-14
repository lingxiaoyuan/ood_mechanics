import numpy as np
import torch


############################ Dataset from MechanicaL MNIST Collection ############################

#covariate shift dataset from Mechanical MNIST Collection
def env_covariate_shift_MNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/train_s100_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/train_s100_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/train_s90_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/train_s90_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/test_s75_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/test_s75_label.txt')

    test2_input = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/test_s50_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_covariate_shift/test_s50_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32), 
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32), 
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})

    return envs

#mechanism shift dataset from Mechanical MNIST Collection
def env_mechanism_shift_MNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/train_s100_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/train_s100_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/train_s90_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/train_s90_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/test_s25_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/test_s25_label.txt')
    
    test2_input = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/test_s10_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_mechanism_shift/test_s10_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32), 
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32), 
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})
    return envs

#sampling bias dataset from Mechanical MNIST Collection
def env_sampling_bias_MNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/train_r15_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/train_r15_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/train_r2neg_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/train_r2neg_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r5neg_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r5neg_label.txt')
    
    test2_input = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r10neg_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r10neg_label.txt')
    
    test3_input = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r1_input.txt')
    test3_label = np.loadtxt('../data/Mechanical_MNIST_sampling_bias/test_r1_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32),
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32),
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test3_input,dtype = torch.float32),
                 'labels':torch.tensor(test3_label,dtype = torch.float32)})
    
    ## scale input to accelerate convergence 
    for i in range(len(envs)):
        envs[i]['images'] = envs[i]['images']/100.0

    return envs




############################ Dataset from Mechanicam MNIST - EMNIST Letters Collection ##########################

#covariate shift dataset from Mechanical MNIST - EMNIST Letters Collection
def env_covariate_shift_EMNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s100_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s100_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s90_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/train_s90_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s75_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s75_label.txt')

    test2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s50_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_covariate_shift/test_s50_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32), 
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32), 
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})

    return envs

#mechanism shift dataset from Mechanical MNIST - EMNIST Letters Collection
def env_mechanism_shift_EMNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s100_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s100_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s90_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/train_s90_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s25_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s25_label.txt')
    
    test2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s10_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_mechanism_shift/test_s10_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32), 
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32), 
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})
    return envs


#sampling bias dataset from Mechanical MNIST - EMNIST Letters Collection
def env_sampling_bias_EMNIST():
    
    train1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r15_input.txt')
    train1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r15_label.txt')
    
    train2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r2neg_input.txt')
    train2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/train_r2neg_label.txt')
    
    test1_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r5neg_input.txt')
    test1_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r5neg_label.txt')
    
    test2_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r10neg_input.txt')
    test2_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r10neg_label.txt')
    
    test3_input = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r1_input.txt')
    test3_label = np.loadtxt('../data/Mechanical_MNIST_EMNIST_sampling_bias/test_r1_label.txt')

    envs = []

    envs.append({'images':torch.tensor(train1_input,dtype = torch.float32), 
                 'labels':torch.tensor(train1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(train2_input,dtype = torch.float32),
                 'labels':torch.tensor(train2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test1_input,dtype = torch.float32),
                 'labels':torch.tensor(test1_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test2_input,dtype = torch.float32),
                 'labels':torch.tensor(test2_label,dtype = torch.float32)})
    envs.append({'images':torch.tensor(test3_input,dtype = torch.float32),
                 'labels':torch.tensor(test3_label,dtype = torch.float32)})
    
    ## scale input to accelerate convergence 
    for i in range(len(envs)):
        envs[i]['images'] = envs[i]['images']/100.0

    return envs


