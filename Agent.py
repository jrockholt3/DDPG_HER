import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam
import numpy as np
import MinkowskiEngine as ME
from spare_tnsr_replay_buffer import ReplayBuffer
from Networks import Actor as Actor
from Networks import Critic as Critic
import pickle
import gc 
from obj_pred_network import Obj_Pred_Net
from Robot_Env import tau_max,scale 

mse_loss = nn.MSELoss()

def check_memory():
    q = 0
    for obj in gc.get_objects():
        try:  
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                q += 1
        except:
            pass
    return q 

class Agent():
    def __init__(self, env, alpha=0.005,beta=0.01, gamma=.99, n_actions=3, 
                time_d=6, max_size=int(1e6), tau=0.005,
                batch_size=64,noise=.01*tau_max,e=.1,enoise=.1*tau_max,
                top_only=False,transfer=False,actor_name = 'actor',critic_name='critic',
                buff_name='replay_buffer'):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size,n_actions,time_d,file=buff_name)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.e = e
        self.enoise = enoise
        self.max_action = torch.tensor(env.action_space.high, device='cuda')
        self.min_action = torch.tensor(env.action_space.low, device='cuda')
        self.tau = tau
        self.score_avg = 0
        self.best_score = 0

        self.actor = Actor(alpha, 1,n_actions,4,name=actor_name,top_only=top_only)
        self.critic = Critic(beta, 1,4,name=critic_name,top_only=top_only)
        self.target_actor = Actor(alpha, 1,n_actions,4,
                                    name='targ_'+actor_name,top_only=top_only)
        self.target_critic = Critic(beta, 1,4,name='targ_'+critic_name,top_only=top_only)
        # self.actor = Actor(name='simple_actor')
        # self.critic = Critic('simple_critic')
        # self.target_actor = Actor('simple_targ_actor')
        # self.target_critic = Critic('simple_targ_critic')

        # loads the convolutional layers from a pre-trained critic
        if transfer:
            temp = Critic(lr=.001,in_feat=1,D=4,name='PID_crit')
            temp.load_checkpoint()
            # load every layer for critic
            self.critic.conv1.load_state_dict(temp.conv1.state_dict())
            self.critic.conv2.load_state_dict(temp.conv2.state_dict())
            self.critic.conv3.load_state_dict(temp.conv3.state_dict())
            self.critic.conv4.load_state_dict(temp.conv4.state_dict())
            self.critic.norm.load_state_dict(temp.norm.state_dict())
            self.critic.linear.load_state_dict(temp.linear.state_dict())
            self.critic.out.load_state_dict(temp.out.state_dict())
            del temp
            temp = Actor(lr=.001,in_feat=1,n_actions=3,D=4,name='PID_train')
            temp.load_checkpoint()
            # load last layers from PID_train
            self.actor.conv1.load_state_dict(temp.conv1.state_dict())
            self.actor.conv2.load_state_dict(temp.conv2.state_dict())
            self.actor.conv3.load_state_dict(temp.conv3.state_dict())
            self.actor.conv4.load_state_dict(temp.conv4.state_dict())
            self.actor.norm.load_state_dict(temp.norm.state_dict())
            self.actor.linear.load_state_dict(temp.linear.state_dict())
            self.actor.out.load_state_dict(temp.out.state_dict())
            del temp

        if top_only:
            # self.actor.conv1.requires_grad_(False)
            # self.actor.conv2.requires_grad_(False)
            # self.actor.conv3.requires_grad_(False)
            self.critic.conv1.requires_grad_(False)
            self.critic.conv2.requires_grad_(False)
            self.critic.conv3.requires_grad_(False)
            self.actor_optim = NAdam(params=self.actor.parameters(), lr=alpha)
            self.critic_optim = NAdam(params=self.critic.parameters(), lr=beta)
        else: 
            self.actor_optim = NAdam(params=self.actor.parameters(), lr=alpha)
            self.critic_optim = NAdam(params=self.critic.parameters(), lr=beta)

        self.update_network_params(tau=1) # hard copy

    def choose_action(self, state, evaluate=False):
        self.actor.eval()
        # q = check_memory()
        x,jnt_err = self.act_preprocessing(state,single_value=True)
        # print('preprocessing added',check_memory()-q)
        with torch.no_grad():
            action = self.actor.forward(x,jnt_err)
        # print('forward pass added',check_memory()-q)

        if not evaluate:
            e = np.random.random()
            if e <= self.e:
                noise = self.enoise
            else:
                noise = self.noise
            action += torch.normal(torch.zeros_like(action),noise)

        return action
    
    def update_network_params(self, tau=None):
        if tau == None:
            tau = self.tau
        '''
        code not needed 
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        '''
        critic_dict = self.critic.state_dict()
        actor_dict = self.actor.state_dict()
        target_critic_dict = self.target_critic.state_dict()
        target_actor_dict = self.target_actor.state_dict()

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() + \
                                (1-tau)*target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_dict)

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() + \
                                (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_dict)

    def remember(self, state, action, reward, new_state, done,t):
        self.memory.store_transition(state,action,reward,new_state,done,t)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 0.0

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()
        self.actor.train()

        new_x, new_jnt_err = self.act_preprocessing(new_state)
        x, jnt_err, a = self.crit_preprocessing(state, action)

        # target actions
        target_actions = self.target_actor.forward(new_x,new_jnt_err)
        # target critic value - value of next state (next state and next action)
        critic_value_ = self.target_critic.forward(new_x,new_jnt_err,target_actions)
        

        target=[]
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*(1-done[j]))
        target = torch.vstack(target)
        
        # update critic 
        self.critic_optim.zero_grad()
        critic_value = self.critic.forward(x,jnt_err,a)
        # critic_loss = F.l1_loss(critic_value, target)
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        mu = self.actor.forward(x,jnt_err)
        actor_loss = -1*self.critic.forward(x,jnt_err,mu).mean()
        # print('actor_loss',actor_loss)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_network_params()
        self.actor.eval()
        self.critic.eval()
        # print('critic_loss', critic_loss.item())
        return critic_loss.item()

    def learn_from_PID(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        x,jnt_err,action = self.crit_preprocessing(state,action)

        self.actor_optim.zero_grad()
        action_ = self.actor.forward(x,jnt_err)
        actor_loss = F.mse_loss(action_, action)
        actor_loss.backward()
        self.actor_optim.step()

        output = actor_loss.detach().cpu().numpy()
        return output

    def train_critic_PID(self):

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        new_x,new_jnt_err = self.act_preprocessing(new_state)
        x,jnt_err,a = self.crit_preprocessing(state,action)

        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        new_action = self.actor(new_x,new_jnt_err)
        critic_value_ = self.target_critic(new_x,new_jnt_err,new_action)

        target=[]
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*(1-done[j]))
        target = torch.vstack(target)

        self.critic_optim.zero_grad()
        critic_value = self.critic.forward(x,jnt_err,a)
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic_optim.step()
       
        self.update_network_params()
        self.critic.train()
        return critic_loss.item()

    def save_models(self):
        print('saving models')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('loading models')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def load_memory(self):
        self.memory = self.memory.load()



    def act_preprocessing(self, state,single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = state[2]#.clone().detach()
            jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
        else:
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = state[2]#.clone().detach()
            jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda')

        x = ME.SparseTensor(coordinates=coords, features=feats.double(),device='cuda')
        return x, jnt_err

    def crit_preprocessing(self, state, action, single_value=False):
        if single_value:
            coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
            jnt_err = state[2]#.clone().detach()
            jnt_err = torch.tensor(jnt_err,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
            a = torch.tensor(action,dtype=torch.double,device='cuda').view(1,state[2].shape[0])
        else:
            coords,feats = ME.utils.sparse_collate(state[0],state[1])
            jnt_err = state[2]#.clone().detach()
            jnt_err = torch.tensor(state[2],dtype=torch.double,device='cuda')
            # action = action.clone().detach()
            # this line causes the thing not to learn
            a = torch.tensor(action,dtype=torch.double,device='cuda') 

        x = ME.SparseTensor(coordinates=coords, features=feats.double(),device='cuda')
        return x, jnt_err, a