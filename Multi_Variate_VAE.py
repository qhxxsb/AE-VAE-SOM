#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:37:20 2021

@author: Zikai Zhu and Peng Su


Note: 
variable name should use lowercase eg: test_variable
Global name should use Uppercase with slash eg: Draw_overview
function name should use uppercase with slash eg: Draw_Overview
class name should use Uppercase without slash eg:DrawOverview

Plz note that the rules of visualization is shown below:
    1) The Legend is always in lower right.
    2) The font size should be 15.
    3) The marker should be chose from "1", "+", "|", "*"
    4) The reference data should show as solid line 
       The other data should choose other linestyle, such as  '--', ':'
    5) The reference data should be Black.
       The other data shoudl choose color should from "b", "g", "r", "c".
"""


### This version is testing the multivariate autoencoder  ###

### Open question: Do we need to denoise the input signals ###



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from matplotlib.patches import ConnectionPatch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import numpy as np
import copy
import matplotlib.gridspec as gridspec
# import Preprocess_Data as predata
class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, input_dim, code_size=1024, device = 'cuda', latent_dim = 6, alpha=1.0, beta=0.9, gamma=1.8, tau=1.4):
        super(VAE, self).__init__()
        # Define the parameters        
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(input_dim, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc11 = nn.Linear(50, latent_dim)
        self.enc_fc12 = nn.Linear(50, latent_dim)
        
        # Define the decoder architecture
        self.dec_fc = nn.Linear(latent_dim,50)
        # self.dec_fc0 = nn.Linear(200,100)
        self.dec_fc1 = nn.Linear(50,10)
        self.dec_fc2 = nn.Linear(10,input_dim)
        # self.dec_fc3 = nn.Linear(10,1)


    def encode(self, x):
        x = self.relu(self.enc_fc0(x))
        x = self.relu(self.enc_fc1(x))
        return self.relu(self.enc_fc11(x)), self.relu(self.enc_fc12(x))
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # print(mu + eps * std)
        return mu + eps * std
    def decode(self,z):
        x =  self.relu(self.dec_fc(z))
        # x =  self.relu(self.dec_fc0(x))
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
    


def Create_Input(amplitude, sample_n,noise_amp, time):
    
    time_stamp =np.linspace(0,time,sample_n) 
    noise = np.random.normal(0,noise_amp,sample_n)
    y_noise=noise+amplitude*np.sin(time_stamp)  
    
    
    y_true = amplitude*np.sin(time_stamp) 
    y_true = y_true.reshape(-1,1)
    y_noise = y_noise.reshape(-1,1)
    return y_true, y_noise

def Fault_Inject(raw_data, noise_amp):
    noise_data = copy.deepcopy(raw_data) 
    noise_inject = noise_amp*np.random.normal(0,0.5, size=(raw_data.shape[0], raw_data.shape[1]))
    noise_data = noise_data + noise_inject
    return noise_data




def Assign_States(win_map):
### Initialize the states for Markov Chain  ### 
    states_dic = {}
    i = 0
    for keys in win_map:
        if len(win_map[keys]) != 0:
            i = i + 1
            states_dic[keys] = ['S' + str(i), i]
    return states_dic

def Get_State(states_dic,w_list):
### The win map list gotten from SOM trajectory ###
    states_list = []
    for i in range (0,len(w_list)):
        states_list.append(states_dic.get(w_list[i]))
    return states_list


def Markov_Transistion(transitions):
    ### Train the Markov transition matrix, input must be the states. ###
    transitions_list = []
    for i in range (0,len(transitions)):
        transitions_list.append(transitions[i][-1])
    n = 1+ max(transitions_list) #number of states

    transitions_matrix = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions_list,transitions_list[1:]):
        transitions_matrix[i][j] += 1

    #now convert to probabilities:
    for row in transitions_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return np.array(transitions_matrix)

# def Temporal_Alignment(win_map, w_list, time_stamp):
#     states_dic = Assign_States(win_map)
#     states_list = Get_State(states_dic,w_list)
#     # for i in range (0,len(states_list)):
        
  

if __name__ == '__main__':
    som_dim = [4,4]
    batch_size = 64

    label_1, train_1 = Create_Input(5, 1000, 0,50)
    label_2, train_2 = Create_Input(3, 1000, 0,50)   
    label_3, train_3 = Create_Input(1, 1000, 0,50)
    ### train data means containing the noise
    dataset = np.concatenate((train_1,train_2,train_3), axis = 1)
    label = np.concatenate((label_1,label_2,label_3), axis = 1)
    # sc = MinMaxScaler()
    # label_dataset = sc.fit_transform(label)
    # train_dataset = sc.fit_transform(train)
    # split_coefficient = 0.6
    # train_dataset = torch.tensor(dataset[:int(len(dataset)*split_coefficient),:])   
    # test_dataset =  torch.tensor(dataset[int(len(dataset)*split_coefficient):-1,:])  
    # train_label = torch.tensor(label[:int(len(dataset)*split_coefficient),:])
    # test_label =  torch.tensor(label[int(len(train_dataset)*split_coefficient):-1,:]) 
    train_dataset = torch.tensor(dataset)   
    train_label = torch.tensor(label)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_dataset.shape[1]
    vae = VAE(input_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss() 
            
    
    # sc = MinMaxScaler()
    # label_dataset = torch.tensor(sc.fit_transform(label_dataset))
    # train_dataset = torch.tensor(sc.fit_transform(train_dataset))
    x = train_dataset.float()
    y = train_label.float()
    # test_x = test_dataset.float()
    # test_y = test_label.float()
    
    BATCH_SIZE = 50
    EPOCH = 100
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
            loss_1 = loss_func(prediction[:,0], b_y[:,0])
            loss_2 = loss_func(prediction[:,1], b_y[:,1])
            loss_3 = loss_func(prediction[:,2], b_y[:,2])
            loss = loss_1 +loss_2+loss_3     # must be (1. nn output, 2. target)
            # loss = loss_func(prediction, b_y)
            if step % 20 == 0:
                print('%d:loss %f' %(epoch,loss))
                
            loss_list.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    x = x.to(device)
    
    # test_x, test_y = Variable(test_x), Variable(test_y)
    
    
    # prediction, z_e = vae(test_x)
    # data_som = z_e.detach().cpu().numpy()
    # som = MiniSom(som_dim[0], som_dim[1], z_e.shape[1], sigma=1.5, learning_rate=0.5)
    # som.pca_weights_init(data_som)
    # som.train(data_som, 1000, random_order=True, verbose=True)  # random training
    # win_map = som.win_map(data_som)
    
    
    #### Train temporal modeling ###
    dataset_temporal = torch.tensor(dataset)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
     
    prediction, z_e = vae(dataset_temporal)
    z_e_array = z_e.data.cpu().numpy()
    data_som = z_e.detach().cpu().numpy()
    som = MiniSom(som_dim[0], som_dim[1], z_e.shape[1], sigma=1.5, learning_rate=0.5)
    som.pca_weights_init(data_som)
    som.train(data_som, 1000, random_order=True, verbose=True)  # random training
    win_map = som.win_map(data_som)
    states_dic = Assign_States(win_map)
    
    
    
    w_list = []
    for i in range (0,len(z_e_array)):
        w = som.winner(z_e_array[i])
        w_list.append(w)   
    states_list = Get_State(states_dic, w_list)
    transition_matrix = Markov_Transistion(states_list)
    # z_e_array = z_e.data.cpu().numpy()
    # z_e_array = z_e_array[:500,:]
    # ### Downsampling ###
    # z_e_down = z_e_array[0:-1:20]

    # w_list = []
    # for i in range (0,len(z_e_down)):
    #     w = som.winner(z_e_down[i])
    #     w_list.append(w)
    #     plt.plot(w[0]+.5,w[1]+.5, marker = '+', color = 'r')
    # fig, ax = plt.subplots()
    # plt.pcolor(som.distance_map().T, cmap='bone_r')
    # plt.colorbar()
    # for i in range (0,len(w_list)):
    #     if i <= len(w_list) -2:
    #         x_start = w_list[i][0] +0.5
    #         x_end = w_list[i+1][0]+0.5
    #         y_start = w_list[i][1]+0.5
    #         y_end = w_list[i+1][1]+0.5
    #         xyA = (x_start,y_start)
    #         xyB = (x_end,y_end)
    #         coordsA = "data"
    #         coordsB = "data"
    #         con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
    #                       arrowstyle="-|>", shrinkA=5, shrinkB=5,
    #                       mutation_scale=20, fc="w",color = 'green')
    #         ax.plot([x_start, x_end], [y_start,y_end], "o")
    #         ax.add_artist(con)
            
 ### Test a new data containing faults ###            
            
# sampling = 1000
# time_stamp = np.linspace(0, 50 ,sampling)  

# label_1, test_1 = Create_Input(5, sampling, 1,max(time_stamp))
# label_2, test_2 = Create_Input(3, sampling, 2,max(time_stamp))   
# label_3, test_3 = Create_Input(1, sampling, 3,max(time_stamp))

# dataset = np.concatenate((test_1,test_2,test_3), axis = 1)
# test_data = torch.tensor(dataset)
# test_data = test_data.float()
# test_data = Variable(test_data)
# prediction, z_e = vae(test_data)
# z_e_array = z_e.data.cpu().numpy()
# z_e_array = z_e_array[:500,:]
# test_timestamp = time_stamp[:500,]

# z_e_down = z_e_array[0:-1:20]
# test_timestamp = test_timestamp[0:-1:20]
# w_list = []
# for i in range (0,len(z_e_down)):
#     w = som.winner(z_e_down[i])
#     w_list.append(w)
#     plt.plot(w[0]+.5,w[1]+.5, marker = '+', color = 'b')
# for i in range (0,len(w_list)):
#     if i <= len(w_list) -2:
#         x_start = w_list[i][0] +0.5
#         x_end = w_list[i+1][0]+0.5
#         y_start = w_list[i][1]+0.5
#         y_end = w_list[i+1][1]+0.5
#         xyA = (x_start,y_start)
#         xyB = (x_end,y_end)
#         coordsA = "data"
#         coordsB = "data"
#         con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
#                       arrowstyle="-|>", shrinkA=5, shrinkB=5,
#                       mutation_scale=20, fc="w",color = 'red',linestyle ='--')
#         ax.plot([x_start, x_end], [y_start,y_end], "o")
#         ax.add_artist(con)
        
        
#### Next step: Training HMM or Markov Chain ####
# def transition_matrix(transitions, states_dic):
#     shape = len(states_dic)
#     M = np.zeros((shape,shape))


    # for i in range (0,len(transitions)):
    #     count = -1
    #     for j in states_dic.keys():
    #         count = count + 1
    #         if states_dic[j] == transitions[i]:
    #             print (count)
            
#     for (i,j) in zip(T,T[1:]):
#         M[i][j] += 1
#     #now convert to probabilities:
#     for row in M:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row]
#     return M



# def transition_matrix(transitions,states_dic):
#     shape = len(states_dic)
    

#     M = [[0]*shape for _ in range(shape)]
#     for i in range (0,len(transitions)):
#         count = -1
#         for j in states_dic.keys():
#             count = count + 1
#             if states_dic[j] == transitions[i]:
#                 M[count][j] + = 1

#     #now convert to probabilities:
#     for row in M:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row]
#     return M