import torch 
import torch.multiprocessing as mp
from Robot_Env import RobotEnv
from spare_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time
from pathos.threading import ThreadPool
from Networks_HER import Actor
from utils import act_preprocessing, stack_arrays
from torch.distributions import MultivariateNormal
from Robot_Env import tau_max

class ParallelRobotEnv():
    def __init__(self, actor_state_dict, noise=.01*tau_max):
        # super(ParallelRobotEnv,self).__init__()
        env = RobotEnv()
        env,state = env.reset()
        self.env = env
        self.memory = ReplayBuffer(250,jnt_d=3, time_d=6)
        self.actor = Actor(in_feat=1, jnt_dim=3,D=4,name='actor',device='cpu')
        self.actor.load_state_dict(actor_state_dict)
        self.noise = noise
        
    def step(self, action, use_PID=False):
        return self.env.step(action, use_PID=use_PID)
    
    def choose_action(self, x, jnt_pos, jnt_goal):
        with torch.no_grad():
            action = self.actor.forward(x, jnt_pos, jnt_goal)
        action += torch.normal(torch.zeros_like(action),self.noise)
        return action

    def reset(self):
        env, state = self.env.reset()
        self.env = env
        return self, state
        
    def get(self):
        return self.memory

def run_episode(env: ParallelRobotEnv):
    done = False
    env, state = env.reset()
    t = 0
    coord_list = [state[0]]
    feat_list = [state[1]]
    score = 0
    while not done:
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2])
        x, jnt_pos, jnt_goal = act_preprocessing(state_,single_value=True,device=env.actor.device)
        with torch.no_grad():
            action = env.choose_action(x, jnt_pos, jnt_goal)
        new_state, reward, done, info = env.step(action)
        env.memory.store_transition(state, action, reward, new_state, done,t)
        state = new_state
        coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
        t+=1
        score += reward
    return env.memory, score



# workers = [ParallelRobotEnv() for i in range(6)]
# t1 = time()
# pool = ThreadPool()
# # pool = ParallelPool()
# # pool = multiprocessing.Pool(len(workers))
# # pool = ProcessPool()

# mems = pool.map(run_episode,workers)
# t2 = time()
# print('time', t2-t1)

# print(mems)