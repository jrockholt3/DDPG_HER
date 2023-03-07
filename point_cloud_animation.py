import platform
import matplotlib
# matplotlib.use('nbAgg'
# print(platform.system())
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import Robot_Env
from Robot_Env import dt
from Object_v2 import rand_object
from Robot3D import workspace_limits as lims 
from utils import stack_arrays
# Fixing random state for reproducibility
import torch
show_box = False
use_PID = True
num_obj = 1 # must be one 
evaluation = True
actor_name = 'actor'
critic_name = 'critic'

class Box():
    def __init__(self):
        a = .08
        self.x_arr = np.array([-a,a,a,-a,-a,-a,a,a,a,a,a,a,-a,-a,-a,-a])
        self.y_arr = np.array([a,a,-a,-a,a,a,a,a,a,-a,-a,-a,-a,-a,-a,a])
        self.z_arr = np.array([a,a,a,a,a,-a,-a,a,-a,-a,a,-a,-a,a,-a,-a])
        self.pos = np.array([0,0,0])

    def render(self,pos):
        return self.x_arr+pos[0], self.y_arr+pos[1], self.z_arr+pos[2]

def gen_centers(x,y,z):
    centers = []
    n = 5
    vec = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    centers.append(np.array([x[0],y[0],z[0]]))
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[0])

    vec = np.array([x[2]-x[1], y[2]-y[1], z[2]-z[1]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[n])
    
    return np.vstack(centers)


# init figure
fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')

# Fifty lines of random 3-D lines
env = Robot_Env.RobotEnv(num_obj=num_obj)
agent = Agent(env,actor_name=actor_name,critic_name=critic_name,e = .005,enoise=torch.tensor([.5,.1,.1]),noise=.005)
agent.load_models()
env, state = env.reset()
print('start', env.start)
# env.start = np.array([0, np.pi/4, -np.pi/4])
# env.goal = np.array([-3*np.pi/4, np.pi/12, -np.pi/6])
goal = env.goal
dt = Robot_Env.dt
max_vel = .6/1
obj1 = rand_object(dt=dt,max_obj_vel=max_vel)
obj2 = rand_object(dt=dt,max_obj_vel=max_vel)
obj3 = rand_object(dt=dt,max_obj_vel=max_vel)
# obj4 = rand_object(dt=dt,max_obj_vel=max_vel)
# obj5 = rand_object(dt=dt,max_obj_vel=max_vel)
# obj6 = rand_object(dt=dt,max_obj_vel=max_vel)
# env.objs = [obj1, obj2, obj3] #, obj4, obj5, obj6]

x_arr = []
y_arr = []
z_arr = []
# obj1's data
x_arr2,y_arr2,z_arr2 = [],[],[]
# temp = env.robot.forward(th=env.start)
# x_arr.append(temp[0,:])
# y_arr.append(temp[1,:])
# z_arr.append(temp[2,:])
# temp = obj.curr_pos

# x_arr2.append(temp[0])
# y_arr2.append(temp[1])
# z_arr2.append(temp[2])

coord_list = [state[0]]
feat_list = [state[1]]

done = False
score = 0
while not done:
    state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
    action = agent.choose_action(state_,evaluate=evaluation)
    state, reward, done, info = env.step(action, use_PID=use_PID)
    score += reward
    coord_list, feat_list = stack_arrays(coord_list, feat_list, state)


    temp = state[0][:,1:4] #776x4 array
    # temp2 = env.robot.forward(th=env.goal)
    # temp = np.hstack((temp, temp2))
    x_arr.append(temp[:,0])
    y_arr.append(temp[:,1])
    z_arr.append(temp[:,2])
    # centers = gen_centers(temp[0,1:],temp[1,1:],temp[2,1:])
    # temp = []
    # for o in env.objs:
    #     temp.append(o.curr_pos)
    # temp = np.vstack(temp)
    # x_arr2.append(temp[:,0])
    # y_arr2.append(temp[:,1])
    # z_arr2.append(temp[:,2])

print('score ', score) 



line, = ax.plot([],[],[], 'bo', lw=2, alpha=.1) # robot at t
# line2, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-1
# line3, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-2 
# line4, = ax.plot([],[],[], 'ro', lw=10) # obj at t
# line5, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-1
# line6, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-2
# line7, = ax.plot([],[],[], 'k-', alpha=.3) # box

j = int(len(x_arr))
def update(i):
    global j
    # set robot lines
    thisx = x_arr[i]
    thisy = y_arr[i]
    thisz = z_arr[i]

    line.set_data_3d(thisx,thisy,thisz)

    return line


# Setting the axes properties
ax.set_xlim3d([0,120])
ax.set_xlabel('X')

ax.set_ylim3d([0,120])
ax.set_ylabel('Y')

ax.set_zlim3d([0,90])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
if show_box:
    N = x_box[0,:].shape[0]
    speed = dt*10000/2
else:
    N = len(x_arr)
    speed = dt*1000

ani = animation.FuncAnimation(
    fig, update, N, interval=speed, blit=False)

# ani.save('file.gif')

plt.show()