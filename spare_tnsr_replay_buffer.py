import numpy as np
import pickle 
# import torch
# import gc 

# def check_memory():
#     q = 0
#     for obj in gc.get_objects():
#         try:  
#             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 q += 1
#         except:
#             pass
#     return q 

class ReplayBuffer:
    def __init__(self, max_size, jnt_d, time_d, file='replay_buffer_no_obj',dir='buffer'):
        # jnt_d = joint dimensions
        self.mem_size = max_size
        self.mem_cntr = 0
        self.jnt_d = jnt_d
        self.time_d = time_d # this is the # of time steps that will be looked over
        # coord_memory, feat_memory, and jnt_err all define the state 
        self.coord_memory = np.empty(max_size,dtype=np.object)
        self.feat_memory = np.empty(max_size,dtype=np.object)
        self.jnt_pos_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.jnt_goal_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)

        self.new_coord_memory = np.empty(max_size,dtype=np.object)
        self.new_feat_memory = np.empty(max_size,dtype=np.object)
        self.new_jnt_pos_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.new_jnt_goal_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.file = dir + '/' + file
        self.time_step = np.ones(self.mem_size)*np.inf # variable for sampling buffer
                                        # will need to go back 'x' # of time steps
                                        # to create the 4D space tensor
                                        # set to np.inf so that all entries do not 
                                        # have a state stored will have a greater timestep
                                        # and avoid storing empty arrays

    def store_transition(self, state, action, reward, new_state, done, time_step):

        # state is (coord_list, feat_list, jnt_err, time_step)
        ndx = self.mem_cntr % self.mem_size
        # q1 = check_memory()
        self.coord_memory[ndx] = state[0] #.clone().detach().cpu()
        self.feat_memory[ndx] = state[1]
        self.jnt_pos_memory[ndx] = state[2]
        self.jnt_goal_memory[ndx] = state[3]
        self.time_step[ndx] = time_step
        self.new_coord_memory[ndx] = new_state[0]
        self.new_feat_memory[ndx] = new_state[1]
        self.new_jnt_pos_memory[ndx] = new_state[2]
        self.new_jnt_goal_memory[ndx] = new_state[3]
        self.terminal_memory[ndx] = done
        self.action_memory[ndx] = action.detach().cpu().numpy()
        self.reward_memory[ndx] = reward
        self.mem_cntr += 1
        # q2 = check_memory()

        # print('operation added',q2-q1,'tensors')


    def sample_buffer(self, batch_size, everything=False):
        min_mem = min(self.mem_cntr, self.mem_size)
        if not everything:
            batch = np.random.choice(min_mem, batch_size, replace=False)
        else:
            batch = np.arange(min_mem,dtype=np.int32)
        
        coord_batch = []
        feat_batch = []
        new_coord_batch = []
        new_feat_batch = []
        coord_list = [] 
        new_coord_list = []
        feat_list = []
        new_feat_list = []
        jnt_err_batch = []
        new_jnt_err_batch = []
        for b in batch:
            for t in range(self.time_d):
                ndx_i = (b - t) % self.mem_size
                if ndx_i >= 0: # check if ndx is out of range
                    if self.time_step[ndx_i]<=self.time_step[b]: # check if still in same episode
                        c_arr = self.coord_memory[ndx_i].copy()
                        c_arr[:,0] = t
                        coord_list.append(c_arr)
                        c_arr = self.new_coord_memory[ndx_i].copy()
                        c_arr[:,0] = t
                        new_coord_list.append(c_arr)
                        feat_list.append(self.feat_memory[ndx_i])
                        new_feat_list.append(self.new_feat_memory[ndx_i])
                    else:
                        t = self.time_d + 1
                else:
                    t = self.time_d + 1
            
            jnt_pos_batch.append(self.jnt_pos_memory[b])
            new_pos_err_batch.append(self.new_jnt_pos_memory[b])
            coord_batch.append(np.vstack(coord_list))
            feat_batch.append(np.vstack(feat_list))
            new_coord_batch.append(np.vstack(new_coord_list))
            new_feat_batch.append(np.vstack(new_feat_list))
            feat_list = []
            coord_list = []
            new_coord_list = []
            new_feat_list = []

        jnt_err = np.vstack(jnt_err_batch)
        new_jnt_err = np.vstack(new_jnt_err_batch)
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return (coord_batch, feat_batch, jnt_err), actions, \
                rewards, (new_coord_batch, new_feat_batch, new_jnt_err), dones
        
    def save(self):
        print('saving buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'wb') as file:
            pickle.dump(self,file)

    def load(self):
        print('loading buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'rb') as file:
            new_buff = pickle.load(file)

        return new_buff
        # # coord_memory, feat_memory, and jnt_err all define the state 
        # self.coord_memory = new_buff.coord_memory
        # self.feat_memory = new_buff.feat_memory
        # self.jnt_err_memory = new_buff.jnt_err_memory

        # self.new_coord_memory = new_buff.new_coord_memory
        # self.new_feat_memory = new_buff.new_feat_memory
        # self.new_jnt_err_memory = new_buff.new_jnt_err_memory

        # self.action_memory = new_buff.action_memory
        # self.reward_memory = new_buff.reward_memory
        # self.terminal_memory = new_buff.terminal_memory
        # self.time_step = new_buff.time_step 

    def clear(self):
        max_size = self.mem_size
        jnt_d = self.jnt_d
        self.coord_memory = np.empty(max_size,dtype=np.object)
        self.feat_memory = np.empty(max_size,dtype=np.object)
        self.jnt_pos_memory = np.empty(self.mem_size,dtype=np.object)

        self.new_coord_memory = np.empty(max_size,dtype=np.object)
        self.new_feat_memory = np.empty(max_size,dtype=np.object)
        self.new_jnt_pos_memory = np.empty(self.mem_size,dtype=np.object)

        self.action_memory = np.zeros((self.mem_size, jnt_d))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def add_batch_data(self, mems):
        
        for mem in mems:
            for i in range(mem.mem_cntr):
                ndx = self.mem_cntr % self.mem_size
                self.coord_memory[ndx] = mem.coord_memory[i]
                self.feat_memory[ndx] = mem.feat_memory[i]
                self.jnt_pos_memory[ndx] = mem.jnt_pos_memory[i]
                self.jnt_goal_memory[ndx] = mem.jnt_goal_memory[i]
                self.action_memory = mem.action_memory[i]
                self.new_coord_memory[ndx] = mem.new_coord_memory[i]
                self.new_feat_memory[ndx] = mem.new_feat_memory[i]
                self.new_jnt_goal_memory[ndx] = mem.new_jnt_goal_memory[i]
                self.new_jnt_pos_memory[ndx] = mem.new_jnt_pos_memory[i]
                self.time_step[ndx] = mem.time_step[i]
                self.reward_memory[ndx] = mem.reward_memory[i]

                self.mem_cntr += 1