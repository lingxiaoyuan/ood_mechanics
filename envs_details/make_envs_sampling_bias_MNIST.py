import numpy as np
import torch

p = '/projectnb/slme/lxyuan/fullField/data/'  ##this should be the path where you save the sampling pool data
#load the input data and rescale it to 1~100
m_train_input = np.loadtxt('%sMNIST_input_files/mnist_img_train.txt'%p).reshape(-1,28*28)/255.0*(100-1)+1
m_train_psi = np.loadtxt('%sFEA_psi_results_equi/summary_psi_train_all.txt'%p)[:,-1].reshape(-1,1)  
m_test_input = np.loadtxt('%sMNIST_input_files/mnist_img_test.txt'%p).reshape(-1,28*28)/255.0*(100-1)+1
m_test_psi = np.loadtxt('%sFEA_psi_results_equi/summary_psi_test_all.txt'%p)[:,-1].reshape(-1,1)

vm = m_train_input.reshape(-1,28*28).sum(1).mean()
vs = m_train_input.reshape(-1,28*28).sum(1).std()
ym = m_train_psi.mean()
ys = m_train_psi.std()

def calculate_fv(X, vm = round(vm, 2), vs =round(vs, 2)):
    V = X.reshape(-1,28*28).sum(1)
    fv = (V - vm)/vs
    return fv

def calculate_fy(y, ym=round(ym, 2),ys=round(ys, 2)):
    temp = y.reshape(-1)
    fy = (temp - ym)/ys
    return fy

def env_sampling_bias_MNIST():
    envs = []
    #train1
    r=15
    images = m_train_input[::2]
    labels = m_train_psi[::2]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 9800, replace=False, p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    #Train2
    r=-2
    images = m_train_input[1::2]
    labels = m_train_psi[1::2]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 200, replace=False, p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})


    #Test1
    r=-5
    images = m_test_input[::3]
    labels = m_test_psi[::3]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 2000, replace=False,p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    
    #Test2
    r=-10
    images = m_test_input[1::3]
    labels = m_test_psi[1::3]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 2000, replace=False,p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    #Test3
    images = m_test_input[2::3]
    labels = m_test_psi[2::3]
    fv = calculate_fv(images)
    k = np.random.choice(len(images), size = 2000, replace=False)
    
    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    return envs
