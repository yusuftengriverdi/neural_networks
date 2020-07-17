import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imghdr
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def ccuda():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on " + str(device) + ".")
    else:
        device = torch.device("cpu")
        print("running on" + str(device) + ".")
    return device


def count_ccuda():
    return torch.cuda.device_count()
