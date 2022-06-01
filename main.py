import numpy as np
import random
import os
import pickle

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from make_envs import *
from model import *
import argparse
from torch import nn, optim, autograd

# function for training model: save the best model and return the error history
#envs is a list of data environments, in which the first two are training environments and the rest are testing environments. Only the training environments will be used for training.  
def training(flags, envs, h, seed):  
    #calculation of the penalty for IRM algorithm
    #the code for calculating the penalty for IRM is from https://github.com/facebookresearch/InvariantRiskMinimization
    def irm_penalty(env):
        #calculate the penalty: the gradient of the risk towards the dummy classifier w=1 
        def penalty(yhat, y):
            scale = torch.tensor(1.).to(dev).requires_grad_()
            loss = loss_fn(yhat * scale, y)
            grad = autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad**2)
        for env in envs[:2]:
            train_size = int(len(env['images'])*0.8)
            env['penalty'] = penalty(env['yhat'], env['labels'][:train_size])
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
        return train_penalty
    
    #calculation of the penalty for REx algorithm
    def rex_penalty(env):
        #calculate the penalty: variance of the risk across all training environments
        train_penalty = torch.stack([envs[0]['nll'], envs[1]['nll']]).std()**2
        return train_penalty
    
    #calculation of the penalty for IGA algorithm
    def iga_penalty(env):
        for env in envs[:2]:
            train_size = int(len(env['images'])*0.8)
            #calculate the risk gradient of all training environments
            env['grad'] = autograd.grad(loss_fn(env['yhat'], env['labels'][:train_size]), model.parameters(), create_graph=True)[0]
        #calculate the penalty: variance of the risk gradient across all environments
        #other way to calculate variance: 
        #train_penalty = (envs[0]['grad']-envs[1]['grad']).pow(2).mean()/2 
        #train_penalty = torch.stack([envs[0]['grad'], envs[1]['grad']]).var()
        train_penalty = (envs[0]['grad']-envs[1]['grad']).pow(2).sum()/2 
        return train_penalty
    
    #choose a model
    if flags.model == 'mlp':
        model = MLP().to(dev)
    elif flags.model == 'lenet':
        model = LeNet().to(dev)
    
    #choose the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=flags.lr)
    loss_fn = nn.MSELoss(reduction = 'mean')

    min_valid_loss = np.inf
    for step in range(flags.steps):
        #training
        model.train()
        for env in envs[:2]:
            # split the training data to be training set(80%) and validation set(20%)
            train_size = int(len(env['images'])*0.8)
            env['yhat'] = model(env['images'][:train_size])
            env['nll'] = loss_fn(env['yhat'], env['labels'][:train_size])            
        
        #calculate the mean loss across all training environments (ERM term)  
        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        
        if flags.algorithm == 'irm':
            train_penalty = irm_penalty(env)
            
        elif flags.algorithm == 'rex':
            train_penalty = rex_penalty(env)
        
        elif flags.algorithm == 'iga':
            train_penalty = iga_penalty(env)
            
        elif flags.algorithm == 'erm':
            train_penalty = torch.tensor(0.)
        
        loss = train_nll.clone()
        
        # add the penalty term after the anneal iteration, and set the penalty term to be very small before the anneal iteration 
        penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1e-4*flags.penalty_weight)
        # add the penalty term to the loss
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
          # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #validating for every 100 steps
        if step % 100 == 0:
            model.eval() 
            #calculate the validation error
            for env in envs[:2]:
                train_size = int(len(env['images'])*0.8)
                env['v_yhat'] = model(env['images'][train_size:])
                env['v_nll'] = loss_fn(env['v_yhat'], env['labels'][train_size:])
            valid_nll = torch.stack([envs[0]['v_nll'], envs[1]['v_nll']]).mean()
            
            #save the training and validation error for every 100 steps. 
            h = np.concatenate([h, 
                    np.array([np.int32(step),
                    train_penalty.detach().cpu().numpy(),
                    train_nll.detach().cpu().numpy(),
                    valid_nll.detach().cpu().numpy()]).reshape(1, -1)])
            
            #save the model with the lowest validation error after the anneal step
            if step > flags.penalty_anneal_iters and min_valid_loss > valid_nll.detach().cpu().numpy():
                min_valid_loss = valid_nll.detach().cpu().numpy()
                torch.save(model.state_dict(), './best_model/%s_%s_pweight%s_steps%s_ann%s_lr%s_seed%s.pt'%(flags.model,flags.algorithm, flags.penalty_weight, flags.steps, flags.penalty_anneal_iters, flags.lr,seed))   
               
    return h

    
if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('./best_model'):
        os.mkdir('./best_model')    
        
    parser = argparse.ArgumentParser(description='OOD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=3001)
    parser.add_argument('--penalty_anneal_iters', type=int, default=500)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--algorithm', type=str,default='irm')
    parser.add_argument('--model', type=str,default='mlp')
    flags = parser.parse_args()
    
    
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    #print all the parameters
    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    
    ##prepare data
    if not os.path.exists('envs.pkl'):
        #load the covariate shift dataste (other dataset can be found in make_envs.py)
        envs = env_covariate_shift_MNIST() 
        f = open("envs.pkl","wb")
        pickle.dump(envs,f)
        f.close()
    else: 
        #read the saved data
        file_to_read = open("envs.pkl", "rb")
        envs = pickle.load(file_to_read)
    
    #sent data to device(cpu or gpu)
    for i in range(len(envs)):
        envs[i]['images'] = envs[i]['images'].to(dev).reshape(-1,1,28,28)
        envs[i]['labels'] = envs[i]['labels'].to(dev).reshape(-1,1)
    
    h = np.ones((0,4))
    seed = 1
    savepath = 'results/%s_%s_pweight%s_steps%s_ann%s_lr%s_'%(flags.model, flags.algorithm, flags.penalty_weight, flags.steps, flags.penalty_anneal_iters, flags.lr)
    while os.path.exists(savepath+"seed%s.txt"%seed):
        seed = seed+1
    print("Seed", seed)
    #save an empty result file to reserve the file name for a seed. 
    np.savetxt(savepath+'seed%s.txt'%seed,np.ones(1))
    
    #reproductivity
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    h = training(flags, envs,h, seed)
    #save the result file  
    np.savetxt(savepath+'seed%s.txt'%seed,h)
