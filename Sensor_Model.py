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
# import Preprocess_Data as predata
class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, code_size=1024, latent_dim = 64, som_dim = [8,8], 
            batch_size = 64, device = 'cuda', alpha=1.0, beta=0.9, gamma=1.8, tau=1.4):
        super(VAE, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.batch_size = batch_size
        # This variable is needed when working with the GPU
        self.embeddings = nn.Parameter(torch.randn(self.som_dim+[self.latent_dim])).to(device)

        probabilities_raw = torch.zeros(self.som_dim+self.som_dim).to(device)
        probabilities_positive = torch.exp(probabilities_raw)
        probabilities_summed = torch.sum(probabilities_positive, dim=[-1,-2], keepdim=True)
        probabilities_normalized = probabilities_positive / probabilities_summed
        self.probs = nn.Parameter(probabilities_positive/probabilities_summed)

        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(1, 10)
        self.enc_fc1 = nn.Linear(10, 50)
        self.enc_fc11 = nn.Linear(50, self.latent_dim)
        self.enc_fc12 = nn.Linear(50, self.latent_dim)
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

    def get_z_q(self,x):
        z_dist = torch.pow((self.embeddings.unsqueeze(0)-x.unsqueeze(1).unsqueeze(2)),2)  # default [64,8,8,64]
        z_dist_sum = torch.sum(z_dist,dim = -1)        # default [64(batch),8,8]
        self.z_dist_flat = z_dist_sum.view(x.shape[0], -1)  # default [64(batch),64]
        self.n_min = torch.argmin(self.z_dist_flat, dim = -1)    # number of minimum distance 
        n_x = self.n_min // self.som_dim[1]
        n_y = self.n_min % self.som_dim[1]
        #n_location = torch.cat((n_x,n_y),1) # location of minimum distance
        n_stacked = torch.stack([n_x, n_y], dim=1)
        z_q = self.gather_nd(self.embeddings, n_stacked)
        z_q_up, z_q_down, z_q_right, z_q_left = self.neighbors(n_x, n_y)
        z_q_neighbors = torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)
        # print(z_q,n_x)
        return z_q, z_q_neighbors

    def neighbors(self, n_x, n_y):
        x_not_top = n_x < self.som_dim[0] - 1
        x_not_bottom = n_x > 0
        y_not_right = n_y < self.som_dim[1] - 1
        y_not_left = n_y > 0
        
        x_up = torch.where(x_not_top, n_x + 1, n_x)
        x_down = torch.where(x_not_bottom, n_x - 1, n_x)
        y_right = torch.where(y_not_right, n_y + 1, n_y)
        y_left = torch.where(y_not_left, n_y - 1, n_y)
        
        z_q_up = torch.zeros(n_x.shape[0], self.latent_dim).to(device)
        z_q_up_ = self.gather_nd(self.embeddings, torch.stack([x_up, n_y], dim=1))
        z_q_up[x_not_top == 1] = z_q_up_[x_not_top == 1]
        
        z_q_down = torch.zeros(n_x.shape[0], self.latent_dim).to(device)
        z_q_down_ = self.gather_nd(self.embeddings, torch.stack([x_down, n_y], dim=1))
        z_q_down[x_not_bottom == 1] = z_q_down_[x_not_bottom == 1]
        
        z_q_right = torch.zeros(n_x.shape[0], self.latent_dim).to(device)
        z_q_right_ =  self.gather_nd(self.embeddings, torch.stack([n_x, y_right], dim=1))
        z_q_right[y_not_right == 1] == z_q_right_[y_not_right == 1]
        
        z_q_left = torch.zeros(n_x.shape[0], self.latent_dim).to(device)
        z_q_left_ = self.gather_nd(self.embeddings, torch.stack([n_x, y_left], dim=1))
        z_q_left[y_not_left == 1] = z_q_left_[y_not_left == 1]
        
        return z_q_up, z_q_down, z_q_right, z_q_left

    def forward(self, x):
        x = x.to(device)    # This variable is needed when working with the GPU
        mu, logvar = self.encode(x)
        z_e = self.reparameterize(mu, logvar)
        z_q, z_q_neighbors = self.get_z_q(z_e)
        decoder_q = self.decode(z_q)
        decoder_e = self.decode(z_e)
        # x = x.reshape(-1,0)
        # x = torch.squeeze(x)
        return z_e, z_q, z_q_neighbors, decoder_e, decoder_q

    def gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)): 
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def loss_reconstruction(self, x, decoder_e, decoder_q):
        loss_e = self.mse_fun(x, decoder_e)
        loss_q = self.mse_fun(x, decoder_q)
        reconstruction_loss = loss_e + loss_q
        return reconstruction_loss

    def loss_som(self, z_e, z_q_neighbors):
        # z_e = z_e.detach()
        som_loss = torch.mean((z_e.unsqueeze(1) - z_q_neighbors)**2)
        return som_loss

    def loss_probabilities(self):
        """Computes the negative log likelihood loss for the transition probabilities."""
        n_x = self.n_min // self.som_dim[1]
        n_y = self.n_min % self.som_dim[1]
        n_x_old = torch.cat([n_x[0:1], n_x[:-1]], dim=0)
        n_y_old = torch.cat([n_y[0:1], n_y[:-1]], dim=0)
        n_stacked = torch.stack([n_x_old, n_y_old, n_x, n_y], dim=1)
 
        probabilities_raw = self.probs
        probabilities_positive = torch.exp(probabilities_raw)
        probabilities_summed = torch.sum(probabilities_positive, dim=[-1,-2], keepdim=True)
        probabilities_normalized = probabilities_positive / probabilities_summed
        self.probs = nn.Parameter(probabilities_positive/probabilities_summed)

        transitions_all = self.gather_nd(self.probs, n_stacked)
        prob_loss = -torch.mean(torch.log(transitions_all))
        return prob_loss

    def loss_z_prob(self):
        """Computes the smoothness loss for the transitions given their probabilities."""
        n_x = self.n_min // self.som_dim[1]
        n_y = self.n_min % self.som_dim[1]
        n_x_old = torch.cat([n_x[0:1], n_x[:-1]], dim=0)
        n_y_old = torch.cat([n_y[0:1], n_y[:-1]], dim=0)
        n_stacked_old = torch.stack([n_x_old, n_y_old], dim=1)

        probabilities_raw = self.probs
        probabilities_positive = torch.exp(probabilities_raw)
        probabilities_summed = torch.sum(probabilities_positive, dim=[-1,-2], keepdim=True)
        probabilities_normalized = probabilities_positive / probabilities_summed
        self.probs = nn.Parameter(probabilities_positive/probabilities_summed)

        out_probabilities_old = self.gather_nd(self.probs, n_stacked_old)
        out_probabilities_flat = out_probabilities_old.view(self.n_min.shape[0], -1)
        weighted_z_dist_prob = self.z_dist_flat*out_probabilities_flat
        prob_z_loss = torch.mean(weighted_z_dist_prob)
        return prob_z_loss

    def loss(self, x, z_e, z_q_neighbors, decoder_e, decoder_q):
        reconstruction_loss = self.loss_reconstruction(x, decoder_e, decoder_q)
        # som_loss = self.loss_som(z_e, z_q_neighbors)
        # probabilities_loss = self.loss_probabilities()
        # probabilities_z_loss = self.loss_z_prob()
        # loss = reconstruction_loss + self.beta*som_loss + self.gamma*probabilities_loss + self.tau*probabilities_z_loss
        loss = reconstruction_loss
        return loss

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
    
if __name__ == '__main__':
    learning_rate=1e-4
    decay_factor = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(device = device).to(device)

    model_param_list = nn.ParameterList()
    for p in vae.named_parameters():
        if p[0] != 'probs':
            model_param_list.append(p[1])
    probs_param_list = nn.ParameterList()
    for p in vae.named_parameters():
        if p[0] == 'probs':
            probs_param_list.append(p[1])

    # optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    opt_model = torch.optim.Adam(model_param_list, lr=learning_rate)
    opt_probs = torch.optim.Adam(probs_param_list, lr=learning_rate*100)

    sc_opt_model = torch.optim.lr_scheduler.StepLR(opt_model, 1000, decay_factor)
    sc_opt_probs = torch.optim.lr_scheduler.StepLR(opt_probs, 1000, decay_factor)

    filepath = "data.csv"
    ASCfile=pd.read_csv(filepath,sep='\t')
    # ASCfile.columns = ['Time','Force','Voltage']
    # BATCH_SIZE=10
    train_dataset = torch.tensor(ASCfile['x'].values)
    label_dataset = torch.tensor(ASCfile['y'].values)
    label_dataset = label_dataset*10
    # print(train_dataset,label_dataset)
    sc = MinMaxScaler()
    label_dataset = torch.tensor(sc.fit_transform(label_dataset.reshape(-1,1)))
    train_dataset = torch.tensor(sc.fit_transform(train_dataset.reshape(-1,1)))
    x = train_dataset.reshape(-1,1).float()
    y = label_dataset.reshape(-1,1).float()
    # print(x.shape,y.shape)
    BATCH_SIZE = 32
    EPOCH = 30
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
            opt_model.zero_grad() # clear gradients for this train
            opt_probs.zero_grad()   

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            b_x, b_y = b_x.to(device), b_y.to(device)
            # print (b_x)
            z_e, __, z_q_neighbors, decoder_e, decoder_q = vae(b_x)        # input x and predict based on x
    
            loss = vae.loss(b_y, z_e, z_q_neighbors, decoder_e, decoder_q) 
            probabilities_loss = vae.loss_probabilities()
            if step % 20 == 0:
                print('%d:loss %f' %(epoch,loss))
            
            loss_list.append(loss.item())

            loss.backward()         # backpropagation, compute gradients
            opt_model.step()        # apply gradients
            probabilities_loss.backward()
            opt_probs.step()
            sc_opt_model.step()
            sc_opt_probs.step()

    plt.figure()
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), color = "blue", alpha=0.2, label = 'Train Data')

    z_e, z_q, z_q_neighbors, decoder_e, prediction = vae(x)     # input x and predict based on x
    # prediction = sc.inverse_transform(prediction)
    # plt.ylim(0,1)
    plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), color='green', alpha=0.5, label = 'Predicted Data')
    plt.xlabel('X')
    averger_mse = sum(loss_list)/len(loss_list)
    plt.title("MES:%f"%(averger_mse))
    plt.ylabel('sin+noise')
    plt.legend(loc = 'best')
    plt.show()
