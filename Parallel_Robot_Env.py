import torch 
import torch.multiprocessing as mp
from Robot_Env import RobotEnv
from spare_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time
from pathos.threading import ThreadPool
from PPO_Networks import Actor
# from Reduced_Networks import Actor
from utils import act_preprocessing, stack_arrays
from torch.distributions import MultivariateNormal
from Robot_Env import tau_max, t_limit, dt

class ParallelRobotEnv():
    def __init__(self, actor_state_dict, noise=.01*tau_max, use_PID=False):
        # super(ParallelRobotEnv,self).__init__()
        env = RobotEnv(num_obj=5)
        mem_size = int(np.round(2*t_limit/dt))
        env,state = env.reset()
        self.env = env
        self.memory = ReplayBuffer(mem_size,jnt_d=3, time_d=6)
        self.goal_memory = ReplayBuffer(mem_size, jnt_d=3, time_d=6)
        self.actor = Actor(in_feat=1, jnt_dim=3,D=4,name='actor',device='cpu')
        self.actor.load_state_dict(actor_state_dict)
        self.noise = noise
        self.use_PID = use_PID
        
    def step(self, action, use_PID=False):
        return self.env.step(action, use_PID=use_PID)
    
    def choose_action(self, x, jnt_pos, jnt_goal):
        if not self.use_PID:
            with torch.no_grad():
                action = self.actor.forward(x, jnt_pos, jnt_goal)
            action += torch.normal(torch.zeros_like(action),self.noise)
        else: 
            action = torch.zeros(3, device='cuda')
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
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
        x, jnt_pos, jnt_goal = act_preprocessing(state_,single_value=True,device=env.actor.device)
        with torch.no_grad():
            action = env.choose_action(x, jnt_pos, jnt_goal)
        new_state, reward, done, info = env.step(action, use_PID=env.use_PID)
        env.memory.store_transition(state, action, reward, new_state, done,t)
        state = new_state
        coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
        t+=1
        score += reward

    fin_ndx = env.memory.mem_cntr - 10 # incase previous episode ended with safety violation, use a finish goal far away from final position
    goal_ = env.memory.jnt_pos_memory[fin_ndx]
    ndx = 0
    mem_cntr = 0
    while not done:
        reward,done = env.env.reward_replay(env.memory.jnt_pos_memory[ndx], env.memory.new_jnt_pos_memory[ndx],goal_)
        state = (env.memory.coord_memory[ndx], env.memory.feat_memory[ndx], env.memory.jnt_pos_memory[ndx], goal_)
        new_state = (env.memory.new_coord_memory[ndx], env.memory.new_feat_memory[ndx], env.memory.new_jnt_pos_memory[ndx], goal_)
        t = env.memory.time_step[ndx]
        env.goal_memory.store_transition(state, env.memory.action_memory[ndx],reward,new_state,
                                            done, t)

        mem_cntr +=1 
        ndx = (mem_cntr) % env.memory.mem_cntr

    
    return env.memory, env.goal_memory, score



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