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
from scipy.spatial import distance
import torch.utils.data as Data
import math
import torch.optim as optim
from torch.autograd import Variable
from matplotlib.patches import ConnectionPatch
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import copy
import sklearn.preprocessing as preprocess 
from sklearn.preprocessing import MinMaxScaler
# import Preprocess_Data as predata
class VAE(nn.Module):
    # Assuming an input size of 
    def __init__(self, input_dim, code_size=1024, device = 'cuda', latent_dim = 3, alpha=1.0, beta=0.9, gamma=1.8, tau=1.4):
        super(VAE, self).__init__()
        # Define the parameters        
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        # Define the encoder architecture
        self.relu = nn.LeakyReLU()
        self.enc_fc0 = nn.Linear(input_dim, 10)
        self.dropout = nn.Dropout(p=0.2)
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
        # x = self.dropout(x)
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
        # x = self.dropout(x)
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
    

def Read_Csv(file_name,column_name):
    """  
    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    column_name : TYPE
        DESCRIPTION.

    Returns
    -------
    raw_data : TYPE
        DESCRIPTION.

    """
    file_path = '/home/pengsu-workstation/SocketSense/GenerateData/' + file_name
    raw_file=pd.read_csv(file_path)
    raw_data = raw_file[column_name].values
    return raw_data.reshape(-1,1) 

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
            states_dic[keys] = ['S' + str(i), i]
            i = i + 1
    return states_dic

def Get_State(states_dic,w_list):
### The win map list gotten from SOM trajectory ###
    states_list = []
    for i in range (0,len(w_list)):
        states_list.append(states_dic.get(w_list[i]))
    return states_list


def Get_TextCoordinator(x0,y0,x1,y1):
    x_inter = abs(x1- x0)/2
    y_inter = abs(y1 - y0)/2
    if x1 > x0:
        x_inter = x_inter + x0
    else:
        x_inter = x_inter + x1
    if y1>y0:
        y_inter = y_inter + y0
    else :
        y_inter = y_inter + y1
    return x_inter, y_inter

def State_Order(states_dic) :
    state_order = []
    for values in states_dic.values():
        state_order.append(values[0])
    return state_order

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
    transitions_matrix = np.array(transitions_matrix)
    return np.array(transitions_matrix)

# def Temporal_Alignment(win_map, w_list, time_stamp):
#     states_dic = Assign_States(win_map)
#     states_list = Get_State(states_dic,w_list)
#     # for i in range (0,len(states_list)):
        
def Second_Neighbor(z_e_array, som):
    second_bmu_list = []
    for i in range (0,len(z_e_array)):
        second_bmu_dic = {}

        weights = som.get_weights()
        for key in win_map:
            if key != som.winner(z_e_array[i]):
                second_bmu_dic[key] = distance.euclidean(z_e_array[i], weights[key[0]][key[1]])
        minval = min(second_bmu_dic.values())
        res = [k for k, v in second_bmu_dic.items() if v==minval]
        second_bmu_list.append(res)
    return second_bmu_list

def Anomaly_Probability(second_bmu_list, z_e_array, som):
    weights = som.get_weights()
    error_coefficent = []
    p_anaomaly_list = []
    for i in range (0,len(z_e_array)):
        winner_index = som.winner(z_e_array[i])
        second_winner_index = second_bmu_list[i][0]
        error = distance.euclidean(z_e_array[i] , weights[winner_index[0]][winner_index[1]])
        sigma = 0.5 * distance.euclidean(weights[second_winner_index[0]][second_winner_index[1]] ,weights[winner_index[0]][winner_index[1]])
        p_anomaly = 1- math.exp(-((error/sigma)**2))
        p_anaomaly_list.append(p_anomaly)
        error_coefficent.append((error/sigma)**2)
    return p_anaomaly_list, error_coefficent

def Affinity_states(win_map, som,states_dic):
    weights = som.get_weights()
    
    ### Define SMU  ###
    ### An offline(pre-trained) method to collect the SMU ###
    affinity_states_dic = {}
    for key in win_map.keys():
        path_state = weights[key[0]][key[1]]
        state_name = states_dic[key][0]
        affinity_states_dic[state_name] = {}
        for affinity_key in win_map.keys():
            # iteration_dic = {}
            if affinity_key !=  key:
                affinity_state_name = states_dic[affinity_key][0]
                distance_euclidean = distance.euclidean(weights[affinity_key[0]][affinity_key[1]], path_state)
                # iteration_dic[affinity_state_name] = distance_euclidean
                affinity_states_dic[state_name][affinity_state_name] = distance_euclidean
        # affinity_states_dic[state_name] = affinity_states_dic[state_name].reshape(1,-1)
        # affinity_states_dic[state_name] = preprocess.normalize(affinity_states_dic[state_name])
    return affinity_states_dic


def Prob_Function(value):
    sigma = 0.5
    prob = math.exp(-(value)/sigma)
    return prob

                
def Define_Prob(affinity_dic):
#### Normalization affinity distance ####
    affinity_threshold = 0.95
    neigbor_unit = {}
    for key in affinity_dic.keys():
        normalization_factor = max(affinity_dic[key].values()) - min(affinity_dic[key].values())
        aux_factor = min(affinity_dic[key].values())
        neigbor_unit[key] = {}
        for affinity_index in affinity_dic[key].keys():
            affinity_dic[key][affinity_index]  = (affinity_dic[key][affinity_index] -aux_factor)/normalization_factor
            # affinity_dic[key][affinity_index] = math.log10(affinity_dic[key][affinity_index])
            affinity_dic[key][affinity_index] = Prob_Function(affinity_dic[key][affinity_index])
            if affinity_dic[key][affinity_index] >= affinity_threshold:
                neigbor_unit[key][affinity_index] = affinity_dic[key][affinity_index]
            
    return neigbor_unit

def Get_Key(dict, value):
    #### the iput dict is (14,14):['S_x', index_nubmer] ####
    #### Return a key, not a list ####

    return [k for k, v in dict.items() if v[0] == value]


def FP_Rate(input_array, som):
    second_bmu_list = Second_Neighbor(z_e_down, som)
    anaomaly_list ,error_list = Anomaly_Probability(second_bmu_list , z_e_down,som)
    anomaly  = 0
    for i in range (0,len(anaomaly_list)):
        if anaomaly_list[i] >= 0.7: 
            anomaly = anomaly + 1
    anomaly = anomaly/ len(anaomaly_list)
    print  (anomaly)
    
    
def Map_Input(input_data, winmap_list,som_dim):
    feature_map = copy.deepcopy(input_data)
    feature_map = feature_map.sum(axis=1)
    test_map_array = np.zeros((som_dim[0],som_dim[1]))
    for i in range (0,len(winmap_list)):
        x_coordinator = winmap_list[i][0]
        y_coordinator =  winmap_list[i][1]
        test_map_array[x_coordinator][y_coordinator] = test_map_array[x_coordinator][y_coordinator] + feature_map[i]
    sc= MinMaxScaler()
    test_map_array = sc.fit_transform(test_map_array)
    return test_map_array

def Rolling_Window(a, window, axis=0):
####  The inputdata should be numpy     ####
    if axis == 0:
        shape = (a.shape[0] - window +1, window, a.shape[-1])
        strides = (a.strides[0],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif axis==1:
        shape = (a.shape[-1] - window +1,) + (a.shape[0], window) 
        strides = (a.strides[-1],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling

def State_Reduce(states_list,hmm_chain_dic ,state_order):
    reshape_state_list = []
    for i in range (0,len(states_list)):
        input_states = states_list[i][0]
        iteration_list = []
        for key in hmm_chain_dic.keys():
            if input_states != key:
                iteration_list.append(hmm_chain_dic[key][input_states])
        repalce_state = state_order[iteration_list.index(max(iteration_list))]
        reshape_state_list.append([repalce_state,iteration_list.index(max(iteration_list))])
    return reshape_state_list



def Train_VAE(train_dataset,train_label):
    
    train_dataset = torch.tensor(train_dataset)   
    train_label = torch.tensor(train_label)
    input_dim = train_dataset.shape[1]
    vae = VAE(input_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss() 
    x = train_dataset.float()
    y = train_label.float()

    BATCH_SIZE = 100
    EPOCH = 500
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=5,)

    # torch can only train on Variable, so convert them to Variable
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
            if step % 50 == 0:
                print('%d:loss %f' %(epoch,loss))
                
            loss_list.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    x = x.to(device)
    return vae

def Train_SOM(vae,dataset): 
    
    #### Train temporal modeling ###
    dataset_temporal = torch.tensor(dataset)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
     
    prediction, z_e = vae(dataset_temporal)
    z_e_array = z_e.data.cpu().numpy()
    # data_som =  z_e.data.cpu().numpy()
    # data_som = z_e.detach().cpu().numpy()
    som = MiniSom(som_dim[0], som_dim[1], z_e_array.shape[1], sigma=1.2, learning_rate=1.5)
    som.pca_weights_init(z_e_array)
    som.train(z_e_array, 10000, random_order=True, verbose=True)  # random training

    

    win_map = som.win_map(z_e_array)
    
    states_dic = Assign_States(win_map)
    affinity_states_dic = Affinity_states(win_map, som,states_dic) 
    prob_dic = Define_Prob(affinity_states_dic)
    
    ### Temporal Modelling ###
    w_list = []
    for i in range (0,len(z_e_array)):
        w = som.winner(z_e_array[i])
        w_list.append(w)   
        
    states_list = Get_State(states_dic, w_list)

    state_order = State_Order(states_dic)
    
    test_map_array = Map_Input(dataset,w_list,som_dim)
    
    state_order = State_Order(states_dic)
    # for key in prob_dic.keys():
    #     for affinity_key in prob_dic[key].keys():
    #         if affinity_key in state_order:
    #             state_order.remove(affinity_key)
    for key in prob_dic.keys():
        for affinity_key in prob_dic[key].keys():
            if affinity_key in state_order:
                state_order.remove(affinity_key)
    # state_order
    
    hmm_chain_dic ={}
    for i in range (0,len(state_order)):
        path_state = state_order[i]
        hmm_chain_dic[path_state] = {}
        for key in affinity_states_dic.keys():
            if key !=path_state:
                hmm_chain_dic[path_state][key] = affinity_states_dic[key][path_state]
    
    reshape_state_list = State_Reduce(states_list, hmm_chain_dic,state_order)
    transition_matrix = Markov_Transistion(reshape_state_list)
    markov_matrix_df = pd.DataFrame(transition_matrix,columns = state_order,index = state_order)
    return  markov_matrix_df,hmm_chain_dic, test_map_array, som, states_dic,win_map,state_order
    
    
    
    
def Test_VAE(vae, test_data):
    test_data = Rolling_Window(test_data, 10)
    dataset_temporal = torch.tensor(test_data)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
    prediction, z_e = vae(dataset_temporal)
    z_e_array = z_e.data.cpu().numpy()
    prediction_array = prediction.data.cpu().numpy()
    return z_e_array, prediction_array

def Test_SOM(z_e_array, som, states_dic,hmm_chain_dic,state_order,markov_matrix_df,test_map_array):
    z_e_down = z_e_array
    w_list_sample = []
    for i in range (0,len(z_e_down)):
        w = som.winner(z_e_down[i])
        w_list_sample.append(w)
    states_list = Get_State(states_dic,w_list_sample)
    reshape_state_list = State_Reduce(states_list, hmm_chain_dic,state_order)
    # w_list_sample  = reshape_state_list
    fig, ax = plt.subplots()
    plt.pcolor(test_map_array.T, cmap = 'Greys')
    cbar = plt.colorbar()
    cbar.set_label('Pressure ', rotation=270) 
    
    for i in range (0,len(reshape_state_list)):
        reshape_state = [i[0] for i in reshape_state_list]
        if states_list[i][0] not in reshape_state:
            x_start = Get_Key(states_dic,states_list[i][0])[0][0] + 0.5
            y_start  = Get_Key(states_dic,states_list[i][0])[0][1] + 0.5
            x_end = Get_Key(states_dic,reshape_state_list[i][0])[0][0] + 0.5
            y_end = Get_Key(states_dic,reshape_state_list[i][0])[0][1] + 0.5
            x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
            xyA = (x_start,y_start)
            xyB = (x_end,y_end)
            coordsA = "data"
            coordsB = "data"
            affinity = round(hmm_chain_dic[reshape_state_list[i][0]][states_list[i][0]], 2)
            ax.text(x_text, y_text,str(affinity),color = 'red')
            con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="w",color = 'red', linestyle = '--')
            ax.plot([x_start, x_end], [y_start,y_end], "o",color = 'red')
            ax.add_artist(con)
    for i in range (0,len(reshape_state_list)):
        if i <= len(states_list) -2 :
            x_start = Get_Key(states_dic,reshape_state_list[i][0])[0][0] + 0.5
            y_start = Get_Key(states_dic,reshape_state_list[i][0])[0][1] + 0.5
            x_end = Get_Key(states_dic,reshape_state_list[i+1][0])[0][0] + 0.5
            y_end = Get_Key(states_dic,reshape_state_list[i+1][0])[0][1] + 0.5
            x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
            xyA = (x_start,y_start)
            xyB = (x_end,y_end)
            coordsA = "data"
            coordsB = "data"
            prob = round(markov_matrix_df[reshape_state_list[i][0]][reshape_state_list[i+1][0]], 2)
            ax.text(x_text, y_text,str(prob),color = 'green')
            con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
                             arrowstyle="-|>", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc="w",color = 'green', linestyle = ':')
            ax.plot([x_start, x_end], [y_start,y_end], "o")
            ax.add_artist(con)
            # ax.text(x_text, y_text,str(round(prob_dic[dict_key][keys], 2)))            
    # for i in range (0,len(w_list_sample)):
         # if i <= len(w_list_sample) -2:
             
    # for i in range (0,len(w_list_sample)):
    #     if i <= len(w_list_sample) -2:

            
    #         current_key = states_dic[w_list_sample[i]][0]
    #         states_list = Get_State(states_dic,w_list_sample)
    #         next_key = states_dic[w_list_sample[i+1]][0]
    #         [current_key,next_key ]= State_Reduce([current_key,next_key])
    #         # dict_key = states_dic[w_list[i]][0]
    #         x_start = w_list_sample[i][0] +0.5
    #         x_end = w_list_sample[i+1][0]+0.5
    #         y_start = w_list_sample[i][1]+0.5
    #         y_end = w_list_sample[i+1][1]+0.5
    #         xyA = (x_start,y_start)
    #         xyB = (x_end,y_end)
    #         coordsA = "data"
    #         coordsB = "data"
    #         # states_dic[w_list[i]]
    #         # ax.text(x_text, y_text,str(round(prob_dic[dict_key][keys], 2)))
    #         reshape_state_list = State_Reduce(states_list, hmm_chaim_dic)
    # #       w_list_sample  = reshape_state_list
    #         x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
    #         ax.text(x_text, y_text,str(round(markov_matrix_df[next_key][current_key], 2)))
    #         temporal_probability.append(markov_matrix_df[next_key][current_key])
    #         con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
    #                       arrowstyle="-|>", shrinkA=5, shrinkB=5,
    #                       mutation_scale=20, fc="w",color = 'green')
    #         ax.plot([x_start, x_end], [y_start,y_end], "o")
    #         ax.add_artist(con) 
    # return temporal_probability
    # for i in range (0,len(w_list_sample)):
    #     if i <= len(w_list_sample) -2:
    #         dict_key = states_dic[w_list_sample[i]][0]
    #         # count = 0
    #         for keys in prob_dic[dict_key].keys():
    #             # if count <= 5:
    #             # count = count + 1
    #             coordinator = get_key(states_dic,keys)[0]
    #             x_start = w_list_sample[i][0] +0.5
    #             x_end = coordinator[0]+0.5
    #             y_start = w_list_sample[i][1]+0.5
    #             y_end = coordinator[1]+0.5
    #             x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
    #             xyA = (x_start,y_start)
    #             xyB = (x_end,y_end)
    #             coordsA = "data"
    #             coordsB = "data"
    #             ax.text(x_text, y_text,str(round(prob_dic[dict_key][keys], 2)))
    #             con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
    #                         arrowstyle="-|>", shrinkA=5, shrinkB=5,
    #                         mutation_scale=20, fc="w",color = 'red', linestyle = '--')
    #             ax.plot([x_start, x_end], [y_start,y_end], "o")
    #             ax.add_artist(con)
    
    
if __name__ == '__main__':
    
    som_dim = [15,15]

    train_ant = Read_Csv('MutilVariate.csv', 'Ant_dist')
    label_ant = Read_Csv('MutilVariate.csv', 'Ant_dist')
    train_post = Read_Csv('MutilVariate.csv', 'Post_dist')
    label_post = Read_Csv('MutilVariate.csv', 'Post_dist')
    train_lat = Read_Csv('MutilVariate.csv', 'Lat_dist')
    label_lat = Read_Csv('MutilVariate.csv', 'Lat_dist')
    label_med = Read_Csv('MutilVariate.csv', 'Med_dist') 
    train_med = Read_Csv('MutilVariate.csv', 'Med_dist') 
    

    dataset = np.concatenate((train_ant,train_post,train_lat,train_med), axis = 1)
    label = np.concatenate((label_ant,label_post,label_lat,label_med), axis = 1)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = Train_VAE(dataset,label)
    markov_matrix_df,hmm_chain_dic, test_map_array, som, states_dic,win_map,state_order= Train_SOM(vae,dataset)
    z_e_array,prediction_array = Test_VAE(vae, dataset)
    Test_SOM(z_e_array[1], som, states_dic,hmm_chain_dic,state_order,markov_matrix_df,test_map_array)
    # z_e_array, prediction_array = Test_VAE(vae, dataset)
    # Test_SOM(z_e_array, som, states_dic)




    # state_order = State_Order(states_dic)
    # for key in prob_dic.keys():
    #     for affinity_key in prob_dic[key].keys():
    #         if affinity_key in state_order:
    #             state_order.remove(affinity_key)

    # hmm_chaim_dic ={}
    # for i in range (0,len(state_order)):
    #     path_state = state_order[i]
    #     hmm_chaim_dic[path_state] = {}
    #     for key in affinity_states_dic.keys():
    #         if key !=path_state:
    #             hmm_chaim_dic[path_state][key] = affinity_states_dic[key][path_state]


    # reshape_state_list = []
    # for i in range (0,len(states_list)):
    #     input_states = states_list[i][0]
    #     iteration_list = []
    #     for key in hmm_chaim_dic.keys():
    #         if input_states != key:
    #             iteration_list.append(hmm_chaim_dic[key][input_states])
    #     repalce_state = state_order[iteration_list.index(max(iteration_list))]
    #     reshape_state_list.append([repalce_state,iteration_list.index(max(iteration_list))])
        
    # z_e_down = z_e_array[0:20]
    # affinity_states_dic = Affinity_states(som,states_dic)
    # test_dic = affinity_states_dic
    # prob_dic = Define_Prob(test_dic)
    # w_list_sample = []
    # for i in range (0,len(z_e_down)):
    #     w = som.winner(z_e_down[i])
    #     w_list_sample.append(w)
    # states_list = Get_State(states_dic, w_list_sample)
    
    
    
    # fig, ax = plt.subplots()
    # plt.pcolor(test_map_array.T, cmap = 'Greys')
    # cbar = plt.colorbar()
    # cbar.set_label('Pressure ', rotation=270) 
    # for i in range (0,len(w_list_sample)):
    #     if i <= len(w_list_sample) -2:
    #         current_key = states_dic[w_list_sample[i]][0]
    #         next_key = states_dic[w_list_sample[i+1]][0]
    #         # dict_key = states_dic[w_list[i]][0]
    #         x_start = w_list_sample[i][0] +0.5
    #         x_end = w_list_sample[i+1][0]+0.5
    #         y_start = w_list_sample[i][1]+0.5
    #         y_end = w_list_sample[i+1][1]+0.5
    #         xyA = (x_start,y_start)
    #         xyB = (x_end,y_end)
    #         coordsA = "data"
    #         coordsB = "data"
    #         # states_dic[w_list[i]]
    #         # ax.text(x_text, y_text,str(round(prob_dic[dict_key][keys], 2)))
    #         x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
    #         ax.text(x_text, y_text,str(round(markov_matrix_df[next_key][current_key], 2)))
    #         con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
    #                       arrowstyle="-|>", shrinkA=5, shrinkB=5,
    #                       mutation_scale=20, fc="w",color = 'green')
    #         ax.plot([x_start, x_end], [y_start,y_end], "o")
    #         ax.add_artist(con) 
    # for i in range (0,len(w_list_sample)):
    #    if i <= len(w_list_sample) -2:
    #        dict_key = states_dic[w_list_sample[i]][0]
    #        count = 0
    #        for keys in prob_dic[dict_key].keys():
    #            # if count <= 5:
    #             # count = count + 1
    #             coordinator = get_key(states_dic,keys)[0]
    #             x_start = w_list_sample[i][0] +0.5
    #             x_end = coordinator[0]+0.5
    #             y_start = w_list_sample[i][1]+0.5
    #             y_end = coordinator[1]+0.5
    #             x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
    #             xyA = (x_start,y_start)
    #             xyB = (x_end,y_end)
    #             coordsA = "data"
    #             coordsB = "data"
    #             ax.text(x_text, y_text,str(round(prob_dic[dict_key][keys], 2)))
    #             con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
    #                        arrowstyle="-|>", shrinkA=5, shrinkB=5,
    #                        mutation_scale=20, fc="w",color = 'red', linestyle = '--')
    #             ax.plot([x_start, x_end], [y_start,y_end], "o")
    #             ax.add_artist(con)
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
        
        
