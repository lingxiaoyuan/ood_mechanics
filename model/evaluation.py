# this code demonstrate how to evaluate the performance of all four algorithms on covariate shift dataset by calculate the root mean squared error of the predicted value and the ground truth.
# the definition of RMSE can be found in the paper "Out of Distribution Generalization for Problems in Mechanics", Eq.(12) and Eq.(13)

import numpy as np
import pandas as pd
from make_envs import *
from model import *
import torch
from torch import nn
import pickle
from sklearn.metrics import mean_squared_error
dev = "cuda:0" if torch.cuda.is_available() else "cpu"


##load data
envs_covariate = []
file_to_read = open("envs.pkl", "rb")
envs_load = pickle.load(file_to_read)

train1_size = int(len(envs_load[0]['images'])*0.8)
train2_size = int(len(envs_load[1]['images'])*0.8)
train_input = np.concatenate([envs_load[0]['images'][:train1_size],envs_load[1]['images'][:train2_size]])
train_label = np.concatenate([envs_load[0]['labels'][:train1_size],envs_load[1]['labels'][:train2_size]])

valid_input = np.concatenate([envs_load[0]['images'][train1_size:],envs_load[1]['images'][train2_size:]])
valid_label = np.concatenate([envs_load[0]['labels'][train1_size:],envs_load[1]['labels'][train2_size:]])

envs_covariate.append({'images':torch.tensor(train_input,dtype = torch.float32),
                     'labels':torch.tensor(train_label,dtype = torch.float32)})
envs_covariate.append({'images':torch.tensor(valid_input,dtype = torch.float32), 
                     'labels':torch.tensor(valid_label,dtype = torch.float32)})
envs_covariate.append({'images':torch.tensor(envs_load[2]['images'],dtype = torch.float32), 
                     'labels':torch.tensor(envs_load[2]['labels'],dtype = torch.float32)})
envs_covariate.append({'images':torch.tensor(envs_load[3]['images'],dtype = torch.float32), 
                     'labels':torch.tensor(envs_load[3]['labels'],dtype = torch.float32)})
envs_covariate.append({'images':torch.tensor(envs_load[4]['images'],dtype = torch.float32), 
                     'labels':torch.tensor(envs_load[4]['labels'],dtype = torch.float32)})

## this new environments contains the training, validation and testing environments
for i in range(len(envs_covariate)):
    envs_covariate[i]['images'] = envs_covariate[i]['images'].to(dev).reshape(-1,1,28,28)
    envs_covariate[i]['labels'] = envs_covariate[i]['labels'].to(dev).reshape(-1,1)

    
#evaluation function    
def evaluate(model, envs, alg, pweight, ann, seed_n):
    loss_fn = nn.MSELoss(reduction = 'mean')
    if model == 'mlp':
        m = MLP()
    elif model =='lenet':
        m = LeNet()

    seeds = range(seed_n)
        
    for env in envs:
        k=0  
        env['nll'] = []
        env['yhat'] = []
        for seed in seeds:
            m.load_state_dict(torch.load("./best_model/%s_%s_pweight%s_steps50001_ann%s_lr0.001_seed%s.pt"%(model,alg,pweight,ann,seed+1),
                                        map_location=torch.device(dev)))
            m.eval()
            m.to(dev)
            y_hat = m(env['images'])
            env['yhat'].append(y_hat.cpu().detach().numpy())
            #env['nll'] is a list of all predicting error of all seeds on the data from this environments, dim = len(seed)
            env['nll'].append(loss_fn(y_hat, env['labels']).cpu().detach().numpy()) 
        env['yhat_mean'] = np.mean(env['yhat'],axis = 0)
        #nll_yhat_mean is the predicting error of the mean prediction, dim = 1
        env['nll_yhat_mean'] = mean_squared_error(env['yhat_mean'], env['labels'].cpu().detach().numpy())
        env['nll'] = np.transpose(np.array(env['nll']))
    return envs


model_name = 'mlp'
alg = ['erm','irm','rex', 'iga']
x_label = ['ERM', 'IRM', 'REx', 'IGA']
pw = [0.0,0.00001,0.1,0.01]
ann = [0,15000,15000,15000]
seed_n = 15
rmse = [[],[],[],[]]
rmse_weight = [[],[],[],[]]

#Evaluation of all algorithms
for a, p,ann_i in zip(alg,pw,ann):
    envs_eva = evaluate(model = model_name, alg = a,  envs = envs_covariate, pweight = p, ann=ann_i, 
                        seed_n = seed_n)
    for i in range(4):
        rmse_weight[i].append(np.sqrt(envs_eva[i]['nll_yhat_mean']))
        rmse[i].append(np.sqrt(envs_eva[i]['nll']))

rmse_all_weight = {'train':rmse_weight[0], 'valid':rmse_weight[1], 'test1':rmse_weight[2],'test2':rmse_weight[3]}
rmse_all = {'train':rmse[0],'valid':rmse[1], 'test1':rmse[2],'test2':rmse[3], }

##print the prediction error of all algorithms 
df_mean  = pd.DataFrame(index=['ERM','IRM','REx','IGA'], columns = ['train','valid','test1','test2'])
for name in ['train','valid','test1','test2']:
    mean_error = []
    for i in range(4):        
        mean_error.append(rmse_all[name][i].mean())
    df_mean[name] = mean_error
print('the average of the root mean squared error(RMSE) of all models with different initializations')
print(df_mean)

df_weight  = pd.DataFrame(index=['ERM','IRM','REx','IGA'], columns = ['train','valid','test1','test2'])
df_weight['train'] = rmse_weight[0]
df_weight['valid'] = rmse_weight[1]
df_weight['test1'] = rmse_weight[2]
df_weight['test2'] = rmse_weight[3]
print('the root mean squared error(RMSE) of the aggregate mean prediction across all models with different initializations')
print(df_weight)