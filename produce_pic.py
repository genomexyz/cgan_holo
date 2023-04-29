from glob import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import itertools
import os
import time
import matplotlib.pyplot as plt

#setting
height_img = 28
width_img = 28
temp_param = 100
batch_size = 64
lr = 0.0002
train_epoch = 200

class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(3, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = self.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = self.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.relu(self.fc3_bn(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def save_generated_img(arr_img_norm_torch, img_filename):
    img_denorm = arr_img_norm_torch * 255
    img_denorm[img_denorm > 255] = 255
    img_denorm[img_denorm < 0] = 0
    cv2.imwrite(img_filename, img_denorm)

checkpoint = torch.load("model_final_cgan.pt")
G = generator().double()
G.load_state_dict(checkpoint['generator'])

temp = torch.randn(1, temp_param).double()
choose_ohc = torch.Tensor([1,1,1]).double().view(1, -1)
G.eval()
fake_img = G(temp, choose_ohc).detach().view(28, 28).numpy()
save_generated_img(fake_img, 'fake_produced.png')