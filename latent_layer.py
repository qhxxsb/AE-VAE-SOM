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

# import Preprocess_Data as predata


class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, lr=0.05):
        super(EmbeddingLayer, self).__init__()
       
        self.num_embeddings = num_embeddings
        self.emb_dimension = embeddings_dim
        self.lr = lr
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embeddings_dim))

    def forward(self, input, som_dim = [8,8]):
        """
        Input: (batch_size, embedding_size)

        """
        if input.size(1) != self.embeddings.size(1):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                            format(input.size(1), embeddings.size(1)))

        batch_size = input.size(0)

        # Compute L2 distance between z_e and embedding weights
        dist = torch.sum(input ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embeddings ** 2, dim=1) - \
               2 * torch.matmul(input, self.embeddings.t())  # [batch_size, num_embeddings]

        # Get the encoding that has the min distance
        n_min = torch.argmin(dist, dim=1, keepdim=False)  # [batch_size, 1]
        n_min = n_min.t()   # [1, batch_size]
        z_q = self.embeddings[n_min]

        # Get the location of winners(batch)
        n_x = n_min // som_dim[1]
        n_y = n_min % som_dim[1]

        # 判断获胜神经元是否处于四周
        x_not_top = n_x < som_dim[0] - 1
        x_not_bottom = n_x > 0
        y_not_right = n_y < som_dim[1] - 1
        y_not_left = n_y > 0     

        # 获得获胜神经元邻居的位置（二维）
        x_up = torch.where(x_not_top, n_x + 1, n_x)
        x_down = torch.where(x_not_bottom, n_x - 1, n_x)
        y_right = torch.where(y_not_right, n_y + 1, n_y)
        y_left = torch.where(y_not_left, n_y - 1, n_y)   

        # 获得获胜神经元邻居的位置（一维）
        n_min_up = x_up*som_dim[1]+n_y
        n_min_down = x_down*som_dim[1]+n_y
        n_min_right = n_x*som_dim[1]+y_right
        n_min_left = n_x*som_dim[1]+y_left

        # [batch_size, 上下左右] 获胜神经元邻居的权值
        z_q_neighbors = torch.stack([self.embeddings[n_min_up], \
            self.embeddings[n_min_down], self.embeddings[n_min_left], \
            self.embeddings[n_min_right]], dim=1)

        # [num_embeddings，batch_size] 存储输入对应获胜神经元的位置用于更新权值
        z_q_local = torch.zeros(self.num_embeddings, batch_size, device=input.device).scatter_(0, n_min.unsqueeze(0), 1)
        z_q_up = torch.zeros(self.num_embeddings, batch_size, device=input.device).scatter_(0, n_min_up.unsqueeze(0), 1)
        z_q_down = torch.zeros(self.num_embeddings, batch_size, device=input.device).scatter_(0, n_min_down.unsqueeze(0), 1)
        z_q_right = torch.zeros(self.num_embeddings, batch_size, device=input.device).scatter_(0, n_min_right.unsqueeze(0), 1)
        z_q_left = torch.zeros(self.num_embeddings, batch_size, device=input.device).scatter_(0, n_min_left.unsqueeze(0), 1)


        # z_e和z_q的差值
        dw = input - self.embeddings[n_min]  # [batch_size, num_embeddings]
        self.embeddings.data = self.embeddings.data + self.lr * torch.mm(z_q_local, dw)
        self.embeddings.data = self.embeddings.data + 0.5 * self.lr * torch.mm(z_q_up, dw) \
            + 0.5 * self.lr * torch.mm(z_q_down, dw) + 0.5 * self.lr * torch.mm(z_q_right, dw) \
            + 0.5 * self.lr * torch.mm(z_q_left, dw) 

        # # Markov transition model
        # trans_prob = torch.zeros(self.num_embeddings, self.num_embeddings).to(device)
        # n_min_old = torch.cat([n_min[0:1], n_min[:-1]], dim=0)  # [0,0,1, ... ,num-2]
        # n_stacked = torch.stack([n_x_old, n_y_old, n_x, n_y], dim=1)

        # loss function:

        # commitment loss
        commit_loss = F.mse_loss(input, z_q)
        # SOM loss
        input = input.detach()
        som_loss = torch.mean((input.unsqueeze(1) - z_q_neighbors)**2)


        return z_q, commit_loss, som_loss

