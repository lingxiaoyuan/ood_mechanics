import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        lin1 = nn.Linear(28*28, 1024)
        lin2 = nn.Linear(1024, 1024)
        lin3 = nn.Linear(1024, 512)
        lin4 = nn.Linear(512, 64)
        lin5 = nn.Linear(64, 1)
        for lin in [lin1, lin2, lin3, lin4, lin5]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), lin4, nn.ReLU(True), lin5)
            
    def forward(self, input):
        out = input.view(input.shape[0], 28*28)
        out = self._main(out)
        return out
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
