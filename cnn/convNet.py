import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib import style
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler


# neural network with dropout
# neural network with dropout

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(64, 64).view(-1, 1, 64, 64)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # print("5.1:", x.shape)
        self.fc2 = nn.Linear(512, 2)
        # print("5.2:", x.shape)
        self.dropout = nn.AlphaDropout(p=0.3)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print("1:", x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # print("2:", x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # print("3:", x.shape)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            # print(x[0].shape[0] , x[0].shape[1] , x[0].shape[2])
        return x

        '''>>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)'''

    def forward(self, x):
        # print("0:", x.shape)
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        # print("4:", x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        # print("5:", x.shape)
        x = self.fc2(x)
        # print("6:", x.shape)
        x = self.dropout(x)
        # print("7:", x.shape)
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    print("Nice")
