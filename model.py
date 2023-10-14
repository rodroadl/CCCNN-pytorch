'''
model.py

Last edited by: GunGyeom James Kim
Last edited at: Oct 12th, 2023
CS 7180: Advnaced Perception

architecture of CNN proposed in Color Constancy Using CNNs
'''
import torch.nn as nn
import torch.nn.functional as F

class CCCNN(nn.Module):
    def __init__(self):
        '''
        constructor
        
        '''
        super(CCCNN, self).__init__()
        self.conv = nn.Conv2d(3, 240, 1)
        self.fc1 = nn.Linear(3840, 40)
        self.fc2 = nn.Linear(40, 3)

    
    def forward(self, x):
        '''
        Return the output estimated by the network

        Parameters:
            x - 32x32x3 image patch
        Return:
            x - estimated output
        '''
        x = self.conv(x) # 32x32x240
        x = F.max_pool2d(x, kernel_size=8, stride=8) # 4x4x240
        print("after max pool:", x)
        x = x.view(-1, 3840) #3840
        print("after reshape:", x)
        x = self.fc1(x)
        print("after fc1:", x)
        x = F.relu(x)
        print("after ReLU:", x)
        x = self.fc2(x)
        print("after fc2:", x)
        return x

        
        