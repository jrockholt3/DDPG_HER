import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv, tau_max, calc_jnt_err
# import gc 
from time import time 
import pickle

# def check_memory():
#     q = 0
#     for obj in gc.get_objects():
#         try:  
#             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 q += 1
#         except:
#             pass
#     return q 
score_hist_file = 'score_hist_0222'
loss_hist_file = 'loss_hist_0222'
alpha = .0001 # actor lr
beta = .001 # critic lr
top_only = False
transfer = False
load_check_ptn = True
load_memory = True
has_objs = True
num_obj = 3
saved_checkpoint = False
top_only = False

episodes = int(1e4)
best_score = -np.inf
n = 100 # number of episodes to calculate the average score
n_batch = 30 # number of batches to train the networks over per episode
batch_size = 512 # batch size

env = RobotEnv(has_objects=has_objs,num_obj=3)
agent = Agent(env,alpha=alpha,beta=beta,batch_size=batch_size,max_size=int(1e6),
                noise=.001*tau_max,e=.1,enoise=.01*tau_max,
                transfer=transfer,top_only=top_only)

if load_check_ptn:
    agent.load_models()
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
    loss = 0
    env, state = env.reset()
    done = False
    score = 0
    t = 0
    coord_list = []
    feat_list = []
    c_arr = state[0]
    for j in range(6):
        c_arr[:,0] = j
        coord_list.append(c_arr.copy())
        feat_list.append(state[1])
    while not done:
        state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2])
        with torch.no_grad():
            action = agent.choose_action(state_)
        new_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, new_state,done,t)
        score += reward
        state = new_state

        coord_list.insert(0,state[0])
        feat_list.insert(0,state[1])
        coord_list.pop()
        feat_list.pop()
        new_list = []
        for j,c in enumerate(coord_list):
            c[:,0] = j
            new_list.append(c.copy())
        coord_list = new_list.copy()
        del new_list
        t += 1    
        
    for j in range(n_batch):
        loss+=agent.learn()
    
    score_history.append(score)
    loss_hist.append(loss/n_batch)

    if np.mean(score_history[-n:]) > best_score and i > n:
        saved_checkpoint = True
        agent.save_models()
        best_score = np.mean(score_history[-n:])
        file = open('tmp/' + score_hist_file, 'wb')
        pickle.dump(loss_hist,file)
        file = open('tmp/' + loss_hist_file, 'wb')
        pickle.dump(score_history, file)


    print('episode', i, 'train_avg %.2f' %np.mean(score_history[-n:]) \
        ,'final jnt_err', np.round(env.jnt_err,2),'time %.2f' %(time()-t1), 'avg loss', loss/n_batch)


best_score = np.mean(score_history[-n:])
file = open('tmp/' + score_hist_file + '.pkl', 'wb')
pickle.dump(loss_hist,file)
file = open('tmp/' + loss_hist_file + '.pkl', 'wb')
pickle.dump(score_history, file)

if not saved_checkpoint:
    agent.save_models()
# actor = agent.actor
# critic = agent.critic
# targ_actor = agent.target_actor
# targ_critic = agent.target_critic

# actor.name = 'fin_actor'
# critic.name = 'fin_critic'
# targ_actor.name = 'fin_targ_actor'
# targ_critic.name = 'fin_targ_critic'

# actor.save_checkpoint()
# critic.save_checkpoint()
# targ_actor.save_checkpoint()
# targ_critic.save_checkpoint()
    # print('memory allocated 0: %f' %(torch.cuda.memory_allocated(0)))
    # GPUtil.showUtilization()

# fig = plt.figure()
# plt.plot(np.arange(len(score_history)), score_history)
# fig2 = plt.figure()
# plt.plot(np.arange(len(loss_hist)), loss_hist)
# plt.show()