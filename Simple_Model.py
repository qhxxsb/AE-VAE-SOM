#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:37:20 2021
@author: pengsu-workstation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import numpy as np
# import Preprocess_Data as predata
class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, code_size=1024, device = 'cuda', latent_dim = 64, alpha=1.0, beta=0.9, gamma=1.8, tau=1.4):
        super(VAE, self).__init__()
        # Define the parameters        
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(1, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc11 = nn.Linear(50, latent_dim)
        self.enc_fc12 = nn.Linear(50, latent_dim)
        
        # Define the decoder architecture
        self.dec_fc = nn.Linear(latent_dim,200)
        self.dec_fc0 = nn.Linear(200,100)
        self.dec_fc1 = nn.Linear(100,50)
        self.dec_fc2 = nn.Linear(50,1)
        # self.dec_fc3 = nn.Linear(10,1)


    def encode(self, x):
        x = self.relu(self.enc_fc0(x))
        x = self.relu(self.enc_fc1(x))
        return self.enc_fc11(x), self.enc_fc12(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # print(mu + eps * std)
        return mu + eps * std
    def decode(self,z):
        x =  self.relu(self.dec_fc(z))
        x =  self.relu(self.dec_fc0(x))
        x = self.relu(self.dec_fc1(x))
        x = self.relu(self.dec_fc2(x))
        # x = self.relu(self.dec_fc3(x))
        return x
    def forward(self, x):
        x = x.to(device)    # This variable is needed when working with the GPU
        mu, logvar = self.encode(x)
        z_e = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z_e)
        # x = x.reshape(-1,0)
        # x = torch.squeeze(x)
        return decoder_out, z_e
    
class AE(nn.Module):
    def __init__(self, code_size=1024):
        super(AE, self).__init__()
        # Define the encoder architecture
        self.relu = nn.ReLU()
        self.enc_fc0 = nn.Linear(1, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc2 = nn.Linear(50, 100)
        self.enc_fc3 = nn.Linear(100, 50)
        self.enc_fc4 = nn.Linear(50, 10)
        self.enc_fc5 = nn.Linear(10, 1)
    def forward(self, x):
        x =  self.relu(self.enc_fc0(x))
        x =  self.relu(self.enc_fc1(x))
        x =  self.relu(self.enc_fc2(x))
        x =  self.relu(self.enc_fc3(x))
        x =  self.relu(self.enc_fc4(x))
        x =  self.relu(self.enc_fc5(x))
        return x

def latent(latent_dim = 64, som_dim = [8,8], batch_size = 64):
    som = MiniSom(som_dim[0], som_dim[1], sigma=4, learning_rate=0.5)



if __name__ == '__main__':
    latent_dim = 64
    som_dim = [8,8]
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss() 
    filepath = "data.csv"    
    ASCfile=pd.read_csv(filepath,sep='\t')
    # ASCfile.columns = ['Time','Force','Voltage']
    # BATCH_SIZE=10
    train_dataset = torch.tensor(ASCfile['x'].values)
    label_dataset = torch.tensor(ASCfile['y'].values)
    label_dataset = label_dataset*10
    sc = MinMaxScaler()
    label_dataset = torch.tensor(sc.fit_transform(label_dataset.reshape(-1,1)))
    train_dataset = torch.tensor(sc.fit_transform(train_dataset.reshape(-1,1)))
    x = train_dataset.reshape(-1,1).float()
    y = label_dataset.reshape(-1,1).float()
    BATCH_SIZE = 32
    EPOCH = 40
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)
    loss_list = []
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            b_x, b_y = b_x.to(device), b_y.to(device)
            # print (b_x)
            prediction, _ = vae(b_x)     # input x and predict based on x
    
            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            if step % 20 == 0:
                print('%d:loss %f' %(epoch,loss))
                
            loss_list.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    x = x.to(device)
    prediction, z_e = vae(x)
    data_som = z_e.detach().cpu().numpy()
    som = MiniSom(som_dim[0], som_dim[1], latent_dim, sigma=1.5, learning_rate=0.5)
    som.pca_weights_init(data_som)
    som.train(data_som, 1000, random_order=True, verbose=True)  # random training
    
    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data_som])
    label_data_np = label_dataset.cpu().numpy()
    som_sum = np.zeros((som_dim[0], som_dim[1]))
    som_sum_n = np.zeros((som_dim[0], som_dim[1]))
    for counter, loc in enumerate(winner_coordinates):
        som_sum[loc[0]][loc[1]] += label_data_np[counter]
        som_sum_n[loc[0]][loc[1]] += 1
    som_sum = som_sum/som_sum_n


    fig = plt.figure(figsize=(18, 8)) # (width, height) 

    # width_plot = np.ones(som_dim[1]+1)
    # width_plot[0] = som_dim[1]
    # gs = gridspec.GridSpec(nrows=som_dim[0], ncols=som_dim[1]+1, width_ratios=width_plot)
    
    # ax0 = fig.add_subplot(gs[0:, 0])
    ax0 = plt.subplot(121)
    ax0.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue", alpha=0.2, label = 'Train Data')
    ax0.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), color='green', alpha=0.5, label = 'Predicted Data')
    ax0.set_xlabel('X')
    averger_mse = sum(loss_list)/len(loss_list)
    ax0.set_title("MES:%f"%(averger_mse))
    ax0.set_ylabel('sin+noise')
    ax0.legend(loc = 'best')

    ax1 = plt.subplot(122)
    obj = ax1.pcolor(som_sum, cmap='bone_r')
    plt.colorbar(obj, ax=ax1)

    plt.show()