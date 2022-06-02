import numpy as np
import torch

p = '/projectnb/slme/lxyuan/fullField/data/'  ##this should be the path where you save the sampling pool data
el_train_input = np.loadtxt('%sLetters_EMNIST_Equibiaxial_OpenBU/input_data/train_data_emnist_letters.txt'%p).reshape(-1,28*28)/255.0*(100-1)+1
el_test_input = np.loadtxt('%sLetters_EMNIST_Equibiaxial_OpenBU/input_data/test_data_emnist_letters.txt'%p).reshape(-1,28*28)/255.0*(100-1)+1
el_train_psi = np.loadtxt('%sLetters_EMNIST_Equibiaxial_OpenBU/EMNIST_Letters_EE_psi_train.txt'%p).reshape(-1,1)
el_test_psi = np.loadtxt('%sLetters_EMNIST_Equibiaxial_OpenBU/EMNIST_Letters_EE_psi_test.txt'%p).reshape(-1,1)

vm = el_train_input.reshape(-1,28*28).sum(1).mean()
vs = el_train_input.reshape(-1,28*28).sum(1).std()
ym = el_train_psi.mean()
ys = el_train_psi.std()

def calculate_fv(X, vm = round(vm, 2), vs =round(vs, 2)):
    V = X.reshape(-1,28*28).sum(1)
    fv = (V - vm)/vs
    return fv

def calculate_fy(y, ym=round(ym, 2),ys=round(ys, 2)):
    temp = y.reshape(-1)
    fy = (temp - ym)/ys
    return fy

def env_sampling_bias_EMNIST():
    envs = []
    #train1
    r=15
    images = el_train_input[::5]
    #images = images + np.random.normal(loc=0.0, scale=1.0, size=np.prod(images.shape)).reshape(images.shape)
    labels = el_train_psi[::5]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 9800, replace=False, p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    #Train2
    r=-2
    images = el_train_input[1::5]
    #images = images + np.random.normal(loc=0.0, scale=1.0, size=np.prod(images.shape)).reshape(images.shape)
    labels = el_train_psi[1::5]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 200, replace=False, p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})


    #Test1
    r=-5
    images = el_test_input[2::5]
    labels = el_test_psi[2::5]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 2000, replace=False,p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    
    #Test2
    r=-10
    images = el_test_input[3::5]
    labels = el_test_psi[3::5]
    fv = calculate_fv(images)
    fy = calculate_fy(labels)
    p = np.power(np.abs(r), -np.abs(fy - np.sign(r)*fv)*5)
    k = np.random.choice(len(images), size = 2000, replace=False,p = p/sum(p))

    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    #Test3
    images = el_test_input[4::5]
    labels = el_test_psi[4::5]
    fv = calculate_fv(images)
    k = np.random.choice(len(images), size = 2000, replace=False)
    
    envs.append({'images':torch.tensor(images[k].reshape(-1,28,28)/100.0,dtype = torch.float32), 
                 'labels':torch.tensor(labels[k].reshape(-1,1),dtype = torch.float32)})
    return envs
