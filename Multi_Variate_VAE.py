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



from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
from scipy.spatial import distance
import torch.utils.data as Data
import math
import torch.optim as optim
from torch.autograd import Variable
from matplotlib.patches import ConnectionPatch,Circle
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import copy
import random
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
        x = x.to(device)
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
        z = z.to(device)
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
    
class ClusterPoint:
    def __init__(self, x=0, y=0, index=0, center = 0, group=0):
    # x,y is location in SOM
        self.x, self.y, self.index, self.group  = x, y, index, group
        self.center = center


def Hierarchical_Cluster(w_list, second_w_list, win_map, clusters_number):
    """
    grouping the neurons in SOM model based on the similarities of prototypes
    """
    points_number = som_dim[0]*som_dim[1] # initial number of groups  
    clusters = [ClusterPoint() for _ in range(points_number)]
    # Delete neurons with no corresponding input
    need_del = [] 
    for index, point in enumerate(clusters):
        x = index // som_dim[1]
        y = index % som_dim[1]
        key = tuple([x,y])
        if len(win_map[key]) == 0:
            need_del.append(index)
        else: 
            point.x = [x]
            point.y = [y]
            point.index = [index]
            point.center = index
            point.group = index
    for i,index in enumerate(need_del):
        del clusters[need_del[i]-i]

    # similarities of prototypes(normalized)
    similarity_matrix = [[0]*points_number for _ in range(points_number)]
    for (best,second) in zip(w_list, second_w_list):
        best_location = best[0] * som_dim[1] + best[1]
        second_location = second[0] * som_dim[1] + second[1]
        similarity_matrix[best_location][second_location] += 1
    similarity_summed = np.sum(similarity_matrix, axis=-1,keepdims=True)
    similarity_normalized = similarity_matrix / similarity_summed

    while len(clusters) > clusters_number:
        # remain last states of clusters
        clusters_copy = copy.deepcopy(clusters)

        # get pairs of clusters
        pair_list = []
        for i in range(len(clusters)):
            cluster_pair = [i,i]
            for j in range(i + 1, len(clusters)):
                i_index = clusters[i].index
                j_index = clusters[j].index
                
                simi_i_sum, closet = 0, 2
                for i_point in i_index:
                    simi_j_sum = 0
                    for j_point in j_index:
                        simi_j_sum += (similarity_normalized[i_point][j_point] +
                        similarity_normalized[j_point][i_point])
                    simi_j_sum = simi_j_sum / len(j_index)
                    simi_i_sum += simi_j_sum
                similarity_i_j = simi_i_sum / len(i_index)
                distance_i_j = 2-similarity_i_j   

                if distance_i_j < closet:
                    closet = distance_i_j
                    cluster_pair = [i,j]
            pair_list.append(cluster_pair)

        # assign groups based on similarity
        pair_list = np.array(pair_list)
        for pair in pair_list:
            same_second = np.where(pair_list[:,1] == pair[1])
            for i in same_second[0]:
                if clusters[pair_list[i][0]].group != clusters[pair_list[i][1]].group:
                    clusters[pair_list[i][0]].group = clusters[pair[0]].group
            clusters[pair[1]].group = clusters[pair[0]].group

        # Delete neurons in the same group until \
        # the number of neurons is reduced to clusters_number
        need_del = []
        meet_quantity = False
        for j, cluster in enumerate(clusters):
            if ~meet_quantity:
                for i, next in enumerate(clusters):
                    if i > j:
                        if next.group == cluster.group:
                            if (len(clusters) - len(need_del)) <= clusters_number:
                                meet_quantity = True
                                break
                            max_occurrence, center_index = 0, 0
                            for center in cluster.index:
                                if similarity_summed[center] > max_occurrence:
                                    max_occurrence = similarity_summed[center]
                                    center_index = center
                            for center in next.index:
                                if similarity_summed[center] > max_occurrence:
                                    max_occurrence = similarity_summed[center]
                                    center_index = center        
                            cluster.x.extend(next.x)
                            cluster.y.extend(next.y)
                            cluster.index.extend(next.index)
                            cluster.center = center_index
                            need_del.append(i)
                            need_del = sorted(need_del)
                            need_del = list(set(need_del))
            else: cluster.group = clusters_copy[j].group
        for i,index in enumerate(need_del):
            del clusters[index-i]

    return clusters

def Winner_Remap(w_list,centers):
    """
    remap winners to the right group
    """
    remap_winner = []
    winner_index = []
    for local in w_list:
        index = int(centers[local[0]][local[1]])
        winner_index.append(index)
        remap_winner.append((index//som_dim[0],index%som_dim[1]))
    winner_index = sorted(winner_index)
    winner_index = list(set(winner_index))
    return remap_winner, winner_index

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
    file_path = 'D:/Project/python/_AE-VAE-SOM/' + file_name
    raw_file=pd.read_csv(file_path)
    raw_data = raw_file[column_name].values
    return raw_data.reshape(-1,1) 

def Read_Anomaly(file_name):
    file_path = 'D:/Project/python/_AE-VAE-SOM/' + file_name
    raw_file=pd.read_csv(file_path)
    raw_data = raw_file.iloc[:,1:].values
    return raw_data

def Windows_Error(file_name = 'Anomaly Inject Location 0.5.csv'):
    file_path = 'D:/Project/python/_AE-VAE-SOM/' + file_name
    error_loc=pd.read_csv(file_path)
    error_loc = error_loc.iloc[:,1:].values
    error_list = []
    for i in range(0,len(error_loc)-1):
        normal_end = error_loc[i][0]+window_size
        end_point = normal_end if normal_end <= error_loc[i+1][0] else error_loc[i+1][0]
        error_window = list(range(error_loc[i][0],end_point))
        error_list = error_list + error_window
    error_window = list(range(error_loc[-1][0],error_loc[-1][0]+10))
    error_list = error_list + error_window
    return error_list

def Create_Input(amplitude, sample_n,noise_amp, time):
    
    time_stamp =np.linspace(0,time,sample_n) 
    noise = np.random.normal(0,noise_amp,sample_n)
    y_noise=noise+amplitude*np.sin(time_stamp)  
    
    
    y_true = amplitude*np.sin(time_stamp) 
    y_true = y_true.reshape(-1,1)
    y_noise = y_noise.reshape(-1,1)
    return y_true, y_noise

def Fault_Inject(amp = 0.5):
    file_path = 'D:/Project/python/_AE-VAE-SOM/' + 'MutilVariate.csv'
    raw_file=pd.read_csv(file_path)
    raw_data = raw_file.values
    # shape = raw_data.shape
    raw_data = copy.deepcopy(raw_data)
    max_data = [] 
    for i in range(0,raw_data.shape[-1]):
        max_data.append(np.max(raw_data[:,i]))
    max_data = np.array(max_data)
    
    # Spike 
    noise_loc = random.sample(range(10,1271),k=50)
    noise_loc = sorted(noise_loc)
    noise_data = copy.deepcopy(raw_data)
    for loc in noise_loc:    
        noise_inject = amp * np.random.choice([-1,1],size=4) * max_data
        noise_data[loc] = noise_data[loc] + noise_inject
    df_data = pd.DataFrame(noise_data,columns=['new_Ant','new_Post','new_Lat','new_Med'])
    loc_data = pd.DataFrame(noise_loc)
    
    df_data.to_csv('Anomaly_Inject_' + str(amp) + '.csv')
    loc_data.to_csv('Anomaly Inject Location '+ str(amp) + '.csv')
    return noise_data


def Assign_States(win_map):
    """
    Get a state dictionary.
    Initialize the states with SOM position for Markov Chain.
    win_map[(i,j)] is a list with all the input that have been 
    mapped to the position.
    """
    states_dic = {}
    i = 0
    for keys in win_map:
        if len(win_map[keys]) != 0:
            states_dic[keys] = ['S' + str(i), i,keys[0]*som_dim[0]   \
            +keys[1],win_map[keys]]
            i = i + 1
    return states_dic 

def Get_State(states_dic,w_list):
    """
    Get the state list with the order of input.
    The win map list gotten from SOM trajectory.
    """
    states_list = []
    for i in range (0,len(w_list)):
        states_list.append(states_dic.get(w_list[i])[0:1])
    return states_list


def Get_TextCoordinator(x0,y0,x1,y1):
    x_inter = (x1 - x0)*0.7
    y_inter = (y1 - y0)*0.7
    x_inter = x_inter + x0
    y_inter = y_inter + y0

    return x_inter, y_inter

def State_Order(states_dic) :
    """
    Get sequence of state “Si”
    """
    state_order = []
    for values in states_dic.values():
        state_order.append(values[0])
    return state_order

# def Markov_Transistion(transitions):
#     """
#     Train the Markov transition matrix, input is the redused states.
#     """
#     transitions_list = []
#     for i in range (0,len(transitions)):
#         transitions_list.append(transitions[i][-1])
#     n = 1+ max(transitions_list) #number of states

#     transitions_matrix = [[0]*n for _ in range(n)]

#     for (i,j) in zip(transitions_list,transitions_list[1:]):
#         transitions_matrix[i][j] += 1

#     #now convert to probabilities:
#     for row in transitions_matrix:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row]
#     transitions_matrix = np.array(transitions_matrix)
#     return np.array(transitions_matrix)

def Markov_Transistion(states_dic,w_list):
    """
    Train the Markov transition matrix with distance, input is all the states.
    """
    states_list = []
    transitions_list = []
    for i in range (0,len(w_list)):
        states_list.append(states_dic.get(w_list[i])[0:1])
        transitions_list.append(w_list[i][0] * som_dim[0] + w_list[i][1])

    size = som_dim[0] * som_dim[1]
    transitions_matrix = [[0]*size for _ in range(size)]

    for (i,j) in zip(transitions_list,transitions_list[1:]):
        transitions_matrix[i][j] += 1

    #now convert to probabilities:
    transitions_matrix = np.array(transitions_matrix)
    # transitions_matrix = np.exp(transitions_matrix)
    transitions_summed = np.sum(transitions_matrix, axis=-1,keepdims=True)
    for i in range(len(transitions_summed)):
        if transitions_summed[i][0] == 0:
            transitions_summed[i][0] = np.iinfo(np.int32).max
    transitions_normalized = transitions_matrix / transitions_summed
    return transitions_normalized,states_list
    
def Gaussian_Neigh(c, sigma=0.8):
    """
    Returns a Gaussian distance centered in c.[x,y]
    """
    g_dis = np.zeros((1,som_dim[0] * som_dim[1]))
    neigx = np.arange(som_dim[0])
    neigy = np.arange(som_dim[1])  # used to evaluate the neighborhood function
    xx = neigx - neigx[c[0]]
    yy = neigy - neigy[c[1]]
    d = 2*sigma*sigma
    ax = np.exp(-np.power(xx, 2)/d)
    ay = np.exp(-np.power(yy, 2)/d)
    for x in range(som_dim[0]):
        for y in range(som_dim[1]):
            g_dis[0][x * som_dim[0] + y] = ax[x] * ay[y]
    return g_dis  

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
    """
    Get dic of distance between state_name & affinity_state_name
    Define SMU  ###
    An offline(pre-trained) method to collect the SMU
    affinity_states_dic[state_name][affinity_state_name]     
    """
    weights = som.get_weights()
    affinity_states_dic = {}
    for key in win_map.keys():
        path_state = weights[key[0]][key[1]]
        state_name = states_dic[key][0] # Si
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
    """
    Normalization affinity distance and remain similar unit
    neigbor_unit[key][affinity_index]
    """
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
    """
    Add the entire rows of input. 
    Assign values to the corresponding SOM neuron.     
    """
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
    """
    The inputdata should be numpy
    a: dataset
    window: a number
    """
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
    """
    Get max dis on all states (from hmm) 
    [S1：...,S2:...,...]
    """
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
        shuffle=True, num_workers=0,)

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
            loss_4 = loss_func(prediction[:,3], b_y[:,3])
            loss = loss_1 +loss_2+loss_3+loss_4     # must be (1. nn output, 2. target)
            # loss = loss_func(prediction, b_y)
            if step % 50 == 0:
                print('%d:loss %f' %(epoch,loss))
                
            loss_list.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
    return vae

def Train_SOM(vae,dataset): 
    
    # Dataset Of Training SOM Modeling 
    dataset_temporal = torch.tensor(dataset)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
    prediction, z_e = vae(dataset_temporal)

    # Get the expectation & variance of input dataset
    expectation, variance = vae.encode(dataset_temporal)
    z_e_array = z_e.data.cpu().numpy()
    expectation, variance = expectation.data.cpu().numpy(), variance.data.cpu().numpy()
    som_in = np.concatenate((expectation,variance),axis=-1) # input of som

    # Training SOM
    som = MiniSom(som_dim[0], som_dim[1], som_in.shape[1], sigma=1.2, learning_rate=1.5)
    # som.pca_weights_init(som_in)
    som.train(som_in, 5000, random_order=True, verbose=True)  # random training
    
    #A dictionary,wm[(i,j)]:patterns that have been mapped to the position (i,j)
    win_map = som.win_map(som_in) 

    # A dictionary. (x,y):[Si,i,keys[0]*som_dim[0]+keys[1],win_map[keys]]
    states_dic = Assign_States(win_map)
    
    # Get the sequence of winner、second winner & weight
    w_list = []
    second_w_list = []
    weight_seq = []    
    weight = som.get_weights()
    for i in range (0,len(som_in)):
        w,second_w = som.get_winner(som_in[i]) # winner & second winner
        w_list.append(w)   # (x,y)
        second_w_list.append(second_w)
        weight_seq.append(weight[w[0]][w[1]])   
    weight_seq = np.array(weight_seq)
    
    # get clusters_number groups
    clusters = Hierarchical_Cluster(w_list, second_w_list,win_map,clusters_number)
    
    # paint different colors for different groups
    fig, ax = plt.subplots()
    ax.set_xlim(0,8)
    ax.set_ylim(8,0)
    Z = np.zeros((som_dim[0],som_dim[1]))
    centers = np.zeros((som_dim[0],som_dim[1]))
    for color, clus in enumerate(clusters):
        for i in range(len(clus.x)):
            centers[clus.x[i]][clus.y[i]] = clus.center
            Z[clus.x[i]][clus.y[i]] = color        
        ax.text(clus.center%som_dim[1]+0.5,clus.center//som_dim[1]+0.5,\
            clus.center,ha="center", va="center",fontsize=12)
    ax.set_aspect(1)
    ax.pcolormesh(Z,cmap = 'Blues')

    # get the remaped winner list  (x,y),(index)
    remap_winner,remaped_index_list = Winner_Remap(w_list,centers)

    # get the corresponding weights
    weight_seq_remap = []    
    for w in remap_winner:
        weight_seq_remap.append(weight[w[0]][w[1]])  

    # calculate the Markov transition of all states
    transition_matrix,states_list = Markov_Transistion(states_dic, w_list)
    # calculate the Markov transition of remaped groups
    transition_remaped,states_list_remaped = Markov_Transistion(states_dic, remap_winner)

    for i in remaped_index_list[:-1]:
        not_zero = []
        for index, j_column in enumerate(transition_remaped[i]):
            if j_column > 0.01:
                not_zero.append(index)
        for j in not_zero:
            x_start = i%som_dim[1]+0.5
            y_start  = i//som_dim[1]+0.5
            x_end = j%som_dim[1]+0.5
            y_end = j//som_dim[1]+0.5
            x_text, y_text  = Get_TextCoordinator(x_start, y_start, x_end, y_end)
            xyA = (x_start,y_start)
            xyB = (x_end,y_end)            
            float_2 = np.round(transition_remaped[i][j],2)
            if i == j:
                ax.text(x_text+0.1, y_text+0.1,str(float_2),color = 'red')    
                plt.scatter(x_start+0.1,y_start+0.1,c='none',marker='o',edgecolors='k',s=400)
            else:
                ax.text(x_text, y_text,str(float_2),color = 'red') 
                ax.annotate("",xy=xyA,xytext=xyB,size=20,arrowprops= \
                    dict(arrowstyle="-|>",fc="w"))
    # plt.tight_layout()
    # plt.colorbar()
    plt.savefig('image.png',bbox_inches = 'tight',pad_inches = 0.05)
    # plt.show()
    
    df = pd.DataFrame(states_list,columns=['winner'])
    df_temp = pd.DataFrame(states_list_remaped,columns=['winner remapped'])
    df = pd.concat([df,df_temp],axis = 1)
    df.to_csv('states_list.csv')
    return  transition_matrix, states_list, transition_remaped, states_list_remaped, som, \
        states_dic, win_map, weight_seq, weight_seq_remap, w_list, remap_winner,centers


def Plot_Data(dataset,vae,weight_seq,weight_seq_remap,anomaly_data):
    dataset = torch.tensor(dataset)
    dataset = dataset.float()
    dataset =  Variable(dataset)
    prediction, z_e = vae(dataset) 
    prediction = prediction.data.cpu().numpy()

    weight_seq = torch.tensor(weight_seq)
    weight_seq = weight_seq.float()
    weight_seq =  Variable(weight_seq)
    weight_seq_remap = torch.tensor(weight_seq_remap)
    weight_seq_remap = weight_seq_remap.float()
    weight_seq_remap =  Variable(weight_seq_remap)
    pre_z_e = vae.decode(z_e)
    pre_som = vae.decode(weight_seq)
    pre_remap = vae.decode(weight_seq_remap)
    pre_z_e = pre_z_e.data.cpu().numpy()
    pre_som = pre_som.data.cpu().numpy()
    pre_remap = pre_remap.data.cpu().numpy()

    fig, axs = plt.subplots(5, 1)
    axs[0].plot(dataset)
    axs[0].set_ylabel('input data')
    axs[0].grid(True)

    axs[1].plot(pre_z_e)
    axs[1].set_ylabel('reconstruction data')
    axs[1].grid(True)

    axs[2].plot(pre_z_e)
    axs[2].set_ylabel('decoding of z_e')
    axs[2].grid(True)

    axs[3].plot(pre_som)
    axs[3].set_ylabel('decoding of som')
    axs[3].grid(True)

    axs[4].plot(pre_remap)
    axs[4].set_xlabel('time')
    axs[4].set_ylabel('decoding remaped winner')
    axs[4].grid(True)
    plt.savefig('result.png',bbox_inches = 'tight',pad_inches = 0.05)
    # plt.show()

def Plot_Noise(dataset,anomaly_data):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(dataset)
    axs[0].set_ylabel('input data')
    axs[0].grid(True)

    axs[1].plot(anomaly_data)
    axs[1].set_ylabel('anomaly data')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

def Test_VAE(vae, test_data,anomaly_data):
    test_data = Rolling_Window(test_data, window_size)
    dataset_temporal = torch.tensor(test_data)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
    prediction, z_e = vae(dataset_temporal)   #[1271,10,4]
    z_e_array = z_e.data.cpu().numpy()
    prediction_array = prediction.data.cpu().numpy()
    expectation, variance = vae.encode(dataset_temporal)
    expectation, variance = expectation.data.cpu().numpy(), variance.data.cpu().numpy()
    som_in = np.concatenate((expectation,variance),axis=-1)

    anomaly_data = Rolling_Window(anomaly_data, window_size)
    dataset_temporal = torch.tensor(anomaly_data)
    dataset_temporal = dataset_temporal.float()
    dataset_temporal =  Variable(dataset_temporal)
    prediction, z_e = vae(dataset_temporal)   #[1271,10,4]
    z_e_array = z_e.data.cpu().numpy()
    prediction_array = prediction.data.cpu().numpy()
    expectation, variance = vae.encode(dataset_temporal)
    expectation, variance = expectation.data.cpu().numpy(), variance.data.cpu().numpy()
    som_in_noise = np.concatenate((expectation,variance),axis=-1)

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(som_in)
    # axs[0].set_ylabel('exp & var data')
    # axs[0].grid(True)
    # axs[1].plot(som_in_noise)
    # axs[1].set_ylabel('exp & var data with noise')
    # axs[1].grid(True)    
    # plt.show()

    return z_e_array, prediction_array, som_in, som_in_noise

def Test_SOM(som_in, som_in_noise, som, states_dic,transition_matrix,centers,window_error):
    # window_error = Windows_Error()
    shape = som_in_noise.shape  #[1271,10,4]
    normal_shape = som_in.shape[0] + window_size - 1  #[1280,10,4]
    # data = np.arange(0,shape[1])
    # df = pd.DataFrame(data)
    p_list = []
    p_local = []
    true_positive = 0
    false_positive = 0
    true_negative = 0
    for lin in range(0,shape[0]):
        w_list_sample = []
        ano_w_list_sample = []
        p = 1
        if lin%normal_shape > (normal_shape - window_size):
            continue
        for i in range (0,shape[1]):
            w_true = som.winner(som_in[lin%normal_shape][i])
            w_anomaly = som.winner(som_in_noise[lin][i])
            w_list_sample.append(w_true)
            ano_w_list_sample.append(w_anomaly)
            remap_winner_true, _ = Winner_Remap(w_list_sample,centers)
            remap_winner_anomaly, _ = Winner_Remap(ano_w_list_sample,centers)
            if i > 0:
                x = remap_winner_anomaly[-2][0] * som_dim[1] + remap_winner_anomaly[-2][1]
                y = remap_winner_anomaly[-1][0] * som_dim[1] + remap_winner_anomaly[-1][1]
                p = p * transition_matrix[x][y]
        p = -np.log(p)
        p_list.append(p)
        p_local.append(lin+(window_size - 1))
        states_list_normal = Get_State(states_dic,remap_winner_true)
        states_list_anomaly = Get_State(states_dic,remap_winner_anomaly)
        column = str(lin+(window_size - 1))
        # if states_list_normal[-1] != states_list_anomaly[-1]:
        #     df_true = pd.DataFrame(states_list_normal,columns=[column])
        #     df_ano = pd.DataFrame(states_list_anomaly,columns=[column])
        #     df = pd.concat([df,df_true,df_ano],axis = 1)
    p_list = np.array(p_list)
    detect_error = np.where(p_list>threshold)
    detect_normal = np.where(p_list<=threshold)
    for loc in detect_error[0]:
        if (loc+(window_size-1)) in window_error:
            true_negative += 1
    for loc in detect_normal[0]:
        if (loc+(window_size-1)) in window_error:
            false_positive += 1
    true_positive = len(detect_normal[0]) - false_positive
    accuracy = (true_positive + true_negative) / shape[0]
    true_negative_rate = true_negative / len(window_error)
    print(f'Error windows is {len(window_error)},Accuracy is {accuracy},True positive is {true_positive},Ture negative is {true_negative} & {true_negative_rate}')
    df_p = pd.DataFrame(p_list)
    df_p_loc = pd.DataFrame(p_local)
    df_p = pd.concat([df_p,df_p_loc],axis = 1)
    return df_p,accuracy,true_negative_rate
    # for lin in range(0,shape[0]):
    #     w_true = som.winner(som_in[lin])
    #     w_anomaly = som.winner(som_in_noise[lin])
    #     if w_true != w_anomaly:
    #         print(lin)
    #     w_list_sample.append(w_true)
    #     ano_w_list_sample.append(w_anomaly)
  
    #     remap_winner_true, _ = Winner_Remap(w_list_sample,centers)
    #     remap_winner_anomaly, _ = Winner_Remap(ano_w_list_sample,centers)

    # states_list_sample = Get_State(states_dic,remap_winner_true)
    # df_true = pd.DataFrame(states_list_sample)
    # states_list_sample = Get_State(states_dic,remap_winner_anomaly)
    # df_ano = pd.DataFrame(states_list_sample)
    # df = pd.concat([df,df_true,df_ano],axis = 1)

    # for lin in range(0,shape[0]):
    #     w_list_sample = []
    #     ano_w_list_sample = []
    #     column = str(lin)
    #     for i in range (0,shape[1]):
    #         w_true = som.winner(som_in[lin][i])
    #         w_anomaly = som.winner(som_in_noise[lin][i])
    #         w_list_sample.append(w_true)
    #         ano_w_list_sample.append(w_anomaly)
  
    #     remap_winner_true, _ = Winner_Remap(w_list_sample,centers)
    #     remap_winner_anomaly, _ = Winner_Remap(ano_w_list_sample,centers)
    #     if remap_winner_true != remap_winner_anomaly:
    #         print(lin)
    #     states_list_sample = Get_State(states_dic,remap_winner_true)
    #     df_true = pd.DataFrame(states_list_sample,columns=[column])
    #     states_list_sample = Get_State(states_dic,remap_winner_anomaly)
    #     df_ano = pd.DataFrame(states_list_sample,columns=[column])
    #     df = pd.concat([df,df_true,df_ano],axis = 1)
    

if __name__ == '__main__':
    
    som_dim = [8,8]
    clusters_number = 9
    window_size = 10
    threshold = 22
    spike_amp = 0.4

    # points_number = som_dim[0] * som_dim[1]

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

    for _ in range(0,5):
        vae = Train_VAE(dataset,label)
        transition_matrix, states_list, transition_remaped, states_list_remaped, som, \
        states_dic, win_map, weight_seq, weight_seq_remap, w_list, remap_winner,centers = Train_SOM(vae,dataset)
        # Plot_Data(dataset,vae,weight_seq,weight_seq_remap,anomaly_data)
        # Plot_Noise(dataset[160:290],anomaly_data)  

        for thre in range(1,30):
            threshold = thre
            accuracy_aver, true_negative_rate_aver = 0, 0

            data = np.arange(0,1281-window_size)
            df_p = pd.DataFrame(data)
            window_error_list = []
            for i in range(0,20):
                Fault_Inject(amp = spike_amp)
                anomaly_data = Read_Anomaly('Anomaly_Inject_' + str(spike_amp) + '.csv')   
                window_error = Windows_Error('Anomaly Inject Location ' + str(spike_amp) + '.csv')    
                z_e_array,prediction_array,som_in,som_in_noise = Test_VAE(vae, dataset,anomaly_data)
                df_temp,accuracy,true_negative_rate = Test_SOM(som_in, som_in_noise, som, states_dic,transition_remaped,centers,window_error)
                df_p = pd.concat([df_p,df_temp],axis = 1)
                window_error_list = window_error_list + window_error
                accuracy_aver += accuracy
                true_negative_rate_aver += true_negative_rate
            accuracy_aver = accuracy_aver /20
            true_negative_rate_aver = true_negative_rate_aver /20
            df_p.to_csv('p list.csv')
            error = pd.DataFrame(window_error_list)
            error.to_csv('error.csv')
            print('%.2f %d %d %.2f Average TN rate is %.4f,Average accuracy is %.4f' % (spike_amp, window_size,clusters_number,threshold,true_negative_rate_aver,accuracy))
            
            with open('result.txt', 'a') as f:
                reslult_str = f'Amp:{spike_amp} Window:{window_size} Som Num:{clusters_number} Yvzhi:{threshold} TN:{true_negative_rate_aver} Acc:{accuracy}\n'
                f.write(reslult_str)
                f.close


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
