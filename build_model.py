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

class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(3, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        #x = F.leaky_relu(self.fc1_1(input), 0.2)
        #y = F.leaky_relu(self.fc1_2(label), 0.2)
        #x = torch.cat([x, y], 1)
        #x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        #x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        #x = F.sigmoid(self.fc4(x))

        x = self.leakyrelu(self.fc1_1(input))
        #print('cek x', x)
        y = self.leakyrelu(self.fc1_2(label))
        x = torch.cat([x, y], 1)
        x = self.leakyrelu(self.fc2_bn(self.fc2(x)))
        x = self.leakyrelu(self.fc3_bn(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def get_all_arr(all_pic):
    all_data_mat = np.zeros((len(all_pic), height_img, width_img))
    for iter_img in range(len(all_pic)):
        single_img_file = all_pic[iter_img]
        img = cv2.imread(single_img_file, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_data_mat[iter_img] = gray
    return all_data_mat

def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    temp_res = torch.randn(1, temp_param).double()
    ohc_moona = torch.Tensor([1,0,0]).double().view(1, -1)
    ohc_kobokan = torch.Tensor([0,1,0]).double().view(1, -1)
    ohc_ollie = torch.Tensor([0,0,1]).double().view(1, -1)
    #print('cek size temp res', temp_res.size())
    G.eval()
    #print('cek shape input', temp_res.size(), ohc_moona.size())
    fake_moona = G(temp_res, ohc_moona).detach()
    fake_kobokan = G(temp_res, ohc_kobokan).detach()
    fake_ollie = G(temp_res, ohc_ollie).detach()
    G.train()

    fig, ax = plt.subplots(3, 1, figsize=(5, 5))
    #size_figure_grid = 10
    #fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    #for i, j in itertools.product(range(3), range(1)):
    #    ax[i, j].get_xaxis().set_visible(False)
    #    ax[i, j].get_yaxis().set_visible(False)
    
    for i in range(3):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    
    ax[0].cla()
    ax[0].imshow(fake_moona.view(28, 28).numpy(), cmap='gray')

    ax[1].cla()
    ax[1].imshow(fake_kobokan.view(28, 28).numpy(), cmap='gray')

    ax[2].cla()
    ax[2].imshow(fake_ollie.view(28, 28).numpy(), cmap='gray')


    #for k in range(3*1):
    #    i = k // 1
    #    j = k % 1
    #    ax[i, j].cla()
    #    ax[i, j].imshow(test_images[k].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

all_moona_png = glob('moona/*.png')
all_kobokan_png = glob('kobo/*.png')
all_ollie_png = glob('ollie/*.png')

arr_moona = get_all_arr(all_moona_png)
arr_kobokan = get_all_arr(all_kobokan_png)
arr_ollie = get_all_arr(all_ollie_png)

arr_moona = np.reshape(arr_moona, (len(arr_moona), -1))
arr_kobokan = np.reshape(arr_kobokan, (len(arr_kobokan), -1))
arr_ollie = np.reshape(arr_ollie, (len(arr_ollie), -1))

arr_moona /= 255
arr_kobokan /= 255
arr_ollie /= 255

arr_unite_img = np.zeros((len(arr_moona)*3, height_img*width_img))
arr_temp = np.zeros((len(arr_moona)*3, temp_param))
arr_ohc = np.zeros((len(arr_moona)*3,3))
for i in range(len(arr_moona)):
    single_temp = np.random.uniform(size=temp_param)
    arr_unite_img[i*3] = arr_moona[i]
    arr_unite_img[i*3+1] = arr_kobokan[i]
    arr_unite_img[i*3+2] = arr_ollie[i]
    arr_temp[i*3] = single_temp
    arr_temp[i*3+1] = single_temp
    arr_temp[i*3+2] = single_temp
    arr_ohc[i*3] = np.array([1,0,0])
    arr_ohc[i*3+1] = np.array([0,1,0])
    arr_ohc[i*3+2] = np.array([0,0,1])
print(arr_unite_img)
print(np.shape(arr_moona), np.shape(arr_kobokan), np.shape(arr_ollie))
print(np.shape(arr_unite_img), np.shape(arr_temp), np.shape(arr_ohc))

arr_unite_img = torch.from_numpy(arr_unite_img)
arr_temp = torch.from_numpy(arr_temp)
arr_ohc = torch.from_numpy(arr_ohc)

dataset_train = TensorDataset(arr_unite_img, arr_temp, arr_ohc)
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# network
G = generator().double()
D = discriminator().double()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('res'):
    os.mkdir('res')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
err_gen = []
err_dis = []
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    G_loss_run = 0.0
    D_loss_run = 0.0

    # learning rate decay
    if (epoch+1) == 30:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 40:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()
    for i, (img, temp, ohc) in enumerate(dataloader):
        #D.zero_grad()
        one_labels = torch.ones(batch_size, 1).double()
        zero_labels = torch.zeros(batch_size, 1).double()

        #print('cek size temp', temp.size())
        #exit()

        #print('cek type', temp.dtype, G.fc1_1.weight.dtype)
        fake_img = G(temp, ohc)
        fake_img = fake_img.detach()

        eval_fake = D(fake_img, ohc)
        eval_real = D(img, ohc)

        D_real_loss = F.binary_cross_entropy(eval_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(eval_fake, zero_labels)
        D_loss = D_real_loss + D_fake_loss

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        fake_img = G(temp, ohc)
        eval_fake = D(fake_img, ohc)
        G_loss = F.binary_cross_entropy(eval_fake, one_labels)

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
    print('Epoch:{},   G_loss:{},    D_loss:{}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
    err_gen.append(G_loss_run/(i+1))
    err_dis.append(D_loss_run/(i+1))
    show_result(epoch, save=True, path='res/res-%s.png'%(epoch))

#save stat
data = {}
data['err_gen'] = np.array(err_gen)
data['err_dis'] = np.array(err_dis)
df = pd.DataFrame(data)
df.to_excel('error.xlsx')

#save model
checkpoint = {'generator': G.state_dict(),
            'discriminator': D.state_dict()}
torch.save(checkpoint, "model_final_cgan.pt")