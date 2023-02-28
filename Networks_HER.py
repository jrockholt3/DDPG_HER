import os 
import torch.nn as nn
from torch.optim import NAdam, Adam
import MinkowskiEngine as ME
import numpy as np
# from Object_v2 import rand_object, Cylinder
import torch
from time import time 
from spare_tnsr_replay_buffer import ReplayBuffer
from Robot_Env import tau_max, scale

conv_out1 = 64
conv_out2 = 128
conv_out4 = 256
linear_out = 512
dropout = 0.2

class Actor(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat, jnt_dim, D, name, chckpt_dir = 'tmp', device='cuda'):
        super(Actor, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg_her')
        self.pool1 = ME.MinkowskiMaxPooling(
            kernel_size=(1,2,2,2),
            stride=(1,2,2,2),
            dimension=4
        )
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=3,
                stride=3,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiBatchNorm(conv_out1).double(),
            ME.MinkowskiReLU().double()
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiBatchNorm(conv_out2).double(),
            ME.MinkowskiReLU().double()
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out4,
                kernel_size=(1,10,10,8),
                stride=(1,10,10,8),
                bias=True,
                dimension=D).double(),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
        )
        self.pooling = ME.MinkowskiGlobalMaxPooling().double()
        self.flatten = ME.MinkowskiToDenseTensor()
        self.norm = nn.Sequential(nn.BatchNorm1d(conv_out4).double(),nn.ReLU().double())
        self.dropout1 = nn.Dropout(dropout).double()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+2*jnt_dim,linear_out).double(),
            nn.BatchNorm1d(linear_out).double(),
            nn.ReLU().double()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(linear_out,jnt_dim,bias=True).double(),
            nn.Tanh().double()
        )

        self.device = device 
        self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def forward(self,x:ME.SparseTensor,jnt_pos,jnt_goal):
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = torch.cat((jnt_pos,jnt_goal,x),dim=1)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.out(x) * tau_max
        return x

    def save_checkpoint(self):
        # print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))

class Critic(ME.MinkowskiNetwork,nn.Module):

    def __init__(self, in_feat, jnt_dim, D, name, chckpt_dir = 'tmp', device='cuda'):
        super(Critic, self).__init__(D)
        self.name = name 
        self.file_path = os.path.join(chckpt_dir,name+'_ddpg_her')

        self.pool1 = ME.MinkowskiMaxPooling(
            kernel_size=(1,2,2,2),
            stride = (1,2,2,2),
            dimension=4
        )
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_feat,
                out_channels=conv_out1,
                kernel_size=3,
                stride=3,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiBatchNorm(conv_out1).double(),
            ME.MinkowskiReLU().double()
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out1,
                out_channels=conv_out2,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=D).double(),
            ME.MinkowskiBatchNorm(conv_out2).double(),
            ME.MinkowskiReLU().double()
        )

        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=conv_out2,
                out_channels=conv_out4,
                kernel_size=(1,10,10,8),
                stride=(1,10,10,8),
                bias=True,
                dimension=D).double(),
            # ME.MinkowskiBatchNorm(conv_out4),
            # ME.MinkowskiSELU()
        )
        self.pooling = ME.MinkowskiGlobalMaxPooling().double()
        self.norm = nn.Sequential(nn.BatchNorm1d(conv_out4).double(),nn.ReLU().double())
        self.dropout1 = nn.Dropout(dropout).double()
        self.linear = nn.Sequential(
            nn.Linear(conv_out4+3*jnt_dim,linear_out).double(),
            nn.BatchNorm1d(linear_out).double(),
            nn.ReLU().double()
        )
        self.dropout2 = nn.Dropout(dropout).double()
        self.out = nn.Sequential(
            nn.Linear(linear_out,1,bias=True).double()
        )
        
        self.device = device
        self.to(self.device)

    def to_dense_tnsr(self, x:ME.SparseTensor):
        y = torch.zeros_like(x.features)
        for c in x.coordinates:
            y[int(c[0])] = x.features[int(c[0])]
        return y

    def forward(self,x:ME.SparseTensor,jnt_pos,jnt_goal,action):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.to_dense_tnsr(x)
        x = self.norm(x)
        x = self.dropout1(x)
        x = torch.cat((jnt_pos,jnt_goal,action,x),dim=1)
        x = self.linear(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x 

    def save_checkpoint(self):
        # print('...saving ' + self.name + '...')
        torch.save(self.state_dict(), self.file_path)

    def load_checkpoint(self):
        print('...loading ' + self.name + '...')
        self.load_state_dict(torch.load(self.file_path))


# batch = 2
# input = torch.ones((batch,1,6,9,120,50))
# x = ME.to_sparse(input,device='cuda',format='BCXXXX')


# actor = Actor(1,n_actions=3, D=4,name='actor')
# y = actor.to_3D(x)
# print(y.coordinates.size())
# print(x.coordinates.size())

# y = actor.forward(x,torch.ones((batch,3),device='cuda'))

# # print(torch.sum(y.coordinates[:,0] - x2.coordinates[:,0]))
# # print(torch.sum(y.coordinates[:,1] - x2.coordinates[:,2]))
# # print(torch.sum(y.coordinates[:,2] - x2.coordinates[:,3]))
# # print(torch.sum(y.coordinates[:,3] - x2.coordinates[:,4]))

