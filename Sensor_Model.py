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
import Preprocess_Data as predata
class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, code_size=1024):
        super(VAE, self).__init__()
        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(1, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc11 = nn.Linear(50, 100)
        self.enc_fc12 = nn.Linear(50, 100)
        self.dec_fc = nn.Linear(100,200)
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        # x = x.reshape(-1,0)
        # x = torch.squeeze(x)
        return decoder_out
    
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
    
# if __name__ == '__main__':
    # vae = AE()
    # optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
    # loss_func = torch.nn.MSELoss() 
    # filepath = "/home/pengsu-workstation/SocketSense/TImeSerialsAnalysis/Raw_data_Sensor_60/SS60 1-1 11,3mm.csv"
    # ASCfile=pd.read_csv(filepath,sep='\t')
    # ASCfile.columns = ['Time','Force','Voltage']
    # # BATCH_SIZE=10
    # train_dataset = torch.tensor(ASCfile['Voltage'].values)
    # label_dataset = torch.tensor(ASCfile['Force'].values)
    # label_dataset = label_dataset*10
    # sc = MinMaxScaler()
    # label_dataset = torch.tensor(sc.fit_transform(label_dataset.reshape(-1,1)))
    # train_dataset = torch.tensor(sc.fit_transform(train_dataset.reshape(-1,1)))
    # x = train_dataset.reshape(-1,1).float()
    # y = label_dataset.reshape(-1,1).float()
    # BATCH_SIZE = 64
    # EPOCH = 100
    # torch_dataset = Data.TensorDataset(x, y)
    # loader = Data.DataLoader(
    #     dataset=torch_dataset, 
    #     batch_size=BATCH_SIZE, 
    #     shuffle=True, num_workers=2,)

    # # torch can only train on Variable, so convert them to Variable
    # x, y = Variable(x), Variable(y)
    # loss_list = []
    # for epoch in range(EPOCH):
    #     for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
    #         b_x = Variable(batch_x)
    #         b_y = Variable(batch_y)
    #         # print (b_x)
    #         prediction = vae(b_x)     # input x and predict based on x
    
    #         loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
    #         if epoch%10 ==0 :
    #             print('loss %f' %loss)
    #             loss_list.append(loss.item())
    #         optimizer.zero_grad()   # clear gradients for next train
    #         loss.backward()         # backpropagation, compute gradients
    #         optimizer.step()        # apply gradients

    # plt.figure()
    # plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2, label = 'Train Data')

    # prediction = vae(x)     # input x and predict based on x
    # # prediction = sc.inverse_transform(prediction)
    # # plt.ylim(0,1)
    # plt.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5, label = 'Predicted Data')
    # plt.xlabel('Voltage')
    # averger_mse = sum(loss_list)/len(loss_list)
    # plt.text(15, 0.2, "MES:%f"%(averger_mse))
    # plt.ylabel('Force')
    # plt.legend(loc = 'best')
    # plt.show()