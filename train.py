import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv, tau_max, calc_jnt_err
from utils import stack_arrays, create_stack
from time import time 
import pickle
from pathos.threading import ThreadPool
from Parallel_Robot_Env import ParallelRobotEnv, run_episode


score_hist_file = 'score_hist_0302'
loss_hist_file = 'loss_hist_0302'
act_name = 'reduced_act'
crit_name = 'reduced_crit'
alpha = .001 # actor lr
beta = .002 # critic lr
gamma = .7
top_only = False
transfer = False
load_check_ptn = False
load_memory = False
has_objs = True
num_obj = 3
top_only = False
epochs = 5 # number of epochs to train over the collected data per episode
num_workers = 10
episodes = 2000
best_score = -np.inf
n = 30 # number of episodes to calculate the average score
batch_size = 512 # batch size

env = RobotEnv(has_objects=has_objs,num_obj=3)
agent = Agent(env,alpha=alpha,beta=beta,batch_size=batch_size,max_size=int(1e6),
                noise=.01*tau_max,e=.1,enoise=.01*tau_max,
                transfer=transfer,top_only=top_only,gamma=gamma,
                actor_name=act_name, critic_name=crit_name)
rng = np.random.default_rng()

if load_check_ptn:
    agent.load_models()
    # file = open('tmp/' + score_hist_file, 'rb')
    # loss_hist = pickle.load(file)
    # file = open('tmp/' + loss_hist_file, 'rb')
    # score_history = pickle.load( file)
    score_history = []
    loss_hist = []
else:
    score_history = []
    loss_hist = []

if load_memory:
    agent.load_memory()

saved_checkpoint = False

for i in range(episodes):
    t1 = time()
    workers = [ParallelRobotEnv(agent.actor.state_dict(),agent.noise) for i in range(num_workers)]
    pool = ThreadPool()
    output = pool.map(run_episode,workers)
    mems,goal_mems,score=[],[],[]
    for tup in output:
        mems.append(tup[0])
        goal_mems.append(tup[1])
        score.append(tup[2])
    score = np.mean(score)

    agent.memory.add_data(mems)
    agent.memory.add_data(goal_mems)

    mem_num = min(agent.memory.mem_cntr,agent.memory.mem_size)
    if mem_num < batch_size:
        batch_size_ = mem_num
        n_batch = 1
    else: 
        n_batch = int(np.floor(mem_num/batch_size))
        batch_size_ = batch_size
    batch = rng.choice(mem_num,size=n_batch*batch_size_,replace=False)

    loss = 0
    for _ in range(epochs):
        for j in range(n_batch):
            loss+=agent.learn(use_batch=True,batch=batch[j*batch_size:(j+1)*batch_size])

    agent.memory.clear()
        
    score_history.append(score)
    loss_hist.append(loss/(epochs*n_batch))

    if np.mean(score_history[-n:]) > best_score and i > n:
        saved_checkpoint = True
        agent.save_models()
        best_score = np.mean(score_history[-n:])
        file = open('tmp/' + score_hist_file+'.pkl', 'wb')
        pickle.dump(loss_hist,file)
        file = open('tmp/' + loss_hist_file+'.pkl', 'wb')
        pickle.dump(score_history, file)

    t2 = time()
    print('episode', i, 'train_avg %.2f' %np.mean(score_history[-n:]) \
        ,'final jnt_err', np.round(env.jnt_err,2),'time %.2f' %(t2-t1), 'avg loss', loss/n_batch)


best_score = np.mean(score_history[-n:])
file = open('tmp/' + score_hist_file + '.pkl', 'wb')
pickle.dump(loss_hist,file)
file = open('tmp/' + loss_hist_file + '.pkl', 'wb')
pickle.dump(score_history, file)

if not saved_checkpoint:
    agent.save_models()
