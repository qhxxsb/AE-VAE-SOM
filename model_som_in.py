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
from latent_layer import EmbeddingLayer
import numpy as np
# import Preprocess_Data as predata


class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, code_size=1024, device = 'cuda', som_dim = [8,8], 
            batch_size = 64, latent_dim = 64, alpha=1.0, beta=0.9, gamma=1.8, tau=1.4):
        super(VAE, self).__init__()
        # Define the parameters        
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.batch_size = batch_size
        
        # This variable is needed when working with the GPU
        self.latent_som = EmbeddingLayer(self.som_dim[0]*self.som_dim[1], self.latent_dim)
        
        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(1, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc11 = nn.Linear(50, self.latent_dim)
        self.enc_fc12 = nn.Linear(50, self.latent_dim)
        
        # Define the decoder architecture
        self.dec_fc = nn.Linear(self.latent_dim,100)
        self.dec_fc0 = nn.Linear(100,60)
        self.dec_fc1 = nn.Linear(60,30)
        self.dec_fc2 = nn.Linear(30,1)
        # self.dec_fc3 = nn.Linear(10,1)
        self.mse_fun = nn.MSELoss()

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
        z_q, commit_loss, som_loss = self.latent_som(z_e, self.som_dim)
    
        decoder_e = self.decode(z_e)
        decoder_q = self.decode(z_q)
        # x = x.reshape(-1,0)
        # x = torch.squeeze(x)
        return z_e, z_q, decoder_e, decoder_q, commit_loss, som_loss

    def loss_reconstruction(self, x, decoder_e, decoder_q):
        loss_e = self.mse_fun(decoder_e, x)
        loss_q = self.mse_fun(decoder_q, x)
        reconstruction_loss = loss_e + loss_q
        return reconstruction_loss

    def loss(self, x, decoder_e, decoder_q, commit_loss, som_loss):
        reconstruction_loss = self.loss_reconstruction(x, decoder_e, decoder_q)
        loss = reconstruction_loss
        return loss

    def gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)): 
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs


if __name__ == '__main__':
    latent_dim = 64
    som_dim = [8,8]
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
    # loss_func = torch.nn.MSELoss() 

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
    EPOCH = 100
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=0)

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)
    loss_list = []
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            b_x, b_y = b_x.to(device), b_y.to(device)
            # print (b_x)
            z_e, z_q, decoder_e, decoder_q, commit_loss, som_loss = vae(b_x)     # input x and predict based on x
    
            loss = vae.loss(b_y, decoder_e, decoder_q, commit_loss, som_loss) 
            if step % 10 == 0:
                print('%d:loss %f' %(epoch,loss))
                
            loss_list.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    plt.figure()
    z_e, z_q, decoder_e, decoder_q, commit_loss, som_loss = vae(x)     # input x and predict based on x
    
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue", alpha=0.2, label = 'Train Data')
    # prediction = sc.inverse_transform(prediction)
    # plt.ylim(0,1)
    plt.plot(x.data.cpu().numpy(), decoder_e.data.cpu().numpy(), color='green', alpha=0.5, label = 'Predicted e')
    plt.plot(x.data.cpu().numpy(), decoder_q.data.cpu().numpy(), color='red', alpha=0.5, label = 'Predicted q')
    
    plt.xlabel('X')
    averger_mse = sum(loss_list)/len(loss_list)
    plt.title("MES:%f"%(averger_mse))
    plt.ylabel('sin+noise')
    plt.legend(loc = 'best')
    plt.show()