import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# train = torchvision.datasets.MNIST('./', download=False)
# test = torchvision.datasets.MNIST('./', train=False, download=False)
# x_train = train.train_data
# y_train = train.train_labels
# x_test = test.data
# y_test = test.targets
# x_train = x_train.float()
# x_test = x_test.float()
# plt.figure(figsize=(10, 10))
# plt.imshow(x_train[0], cmap='Greys')
# plt.show()

# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

#     def forward(self, x):
#         # ParameterList can act as an iterable, or be indexed using ints
#         print(self.params)
#         for i, p in enumerate(self.params):
#             x = self.params[i // 2].mm(x) + p.mm(x)
#         return x


# som_dim=[8,8]
# probs_raw = torch.zeros(som_dim + som_dim)
# print(probs_raw.shape)
# print(probs_raw.shape)
# probs_pos = torch.exp(probs_raw)
# print(probs_pos.shape)
# probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
# print(probs_sum.shape)
# probs = nn.Parameter(probs_pos/probs_sum)
# print(probs.shape)


# x = torch.randn(1,2,3)
# print([8,8] + [64])

# z_e = torch.zeros(8,8,64)
# embedding = torch.ones(64)
# def fun():
#     for i in range(len(embedding)//10):
#         yield embedding[i*10:(i+1)*10]
#     return fun

# print(fun())
# embedding = embedding.unsqueeze(1)
# print(torch.argmin(torch.sum(embedding-z_e, dim=-1),dim = -1))

# rang = torch.ones(1,2,3)
# print(len(rang))


# n_channels=[32, 128]
# flat_size = 4*4*n_channels[-1]
# print(flat_size)

def get_noise_sin(amplitude, sample_n, time, noise_amp):

    x=np.linspace(0,time,sample_n) 
    noise = np.random.normal(0,noise_amp,sample_n)
    y=noise+amplitude*np.sin(x)  
    y_true = amplitude*np.sin(x)  

    plt.plot(x,y,'b-') #绘制成图表
    plt.plot(x,y_true,'r-') #绘制成图表
    
    plt.show()
    data = {'x':x,'y_true':y_true,'y':y}
    df = pd.DataFrame(data)
    #把生成的正弦波数据存入txt文件
    df.to_csv("data.csv",sep='\t')
    ASCfile=pd.read_csv("data.csv",sep='\t')
    print(ASCfile['x'].values)
    # fileH=open("data.txt")
    # fileData=fileH.read()
    # fileH.close()

    # #以写方式打开文件，以之前的两个正弦波的数据做拷贝
    # fileH=open("data-500.txt",'w')
    # for i in range(2000):
    #     fileH.write("\n")
    #     fileH.writelines(fileData)
    # fileH.close()

get_noise_sin(amplitude = 5, sample_n = 1000, time = 10, noise_amp = 0.5)
