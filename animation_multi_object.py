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
use_PID = False
num_obj = 2
evaluation = True
actor_name = 'actor_0308'
critic_name = 'critic_0308'

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
# start = np.array([np.pi/4, np.pi/4, np.pi/4])
# goal = -1*start
# env.goal=goal
# env.robot.set_pose(start)
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


    temp = env.robot.forward()
    temp2 = env.robot.forward(th=env.goal)
    temp = np.hstack((temp, temp2))
    x_arr.append(temp[0,:])
    y_arr.append(temp[1,:])
    z_arr.append(temp[2,:])
    centers = gen_centers(temp[0,1:],temp[1,1:],temp[2,1:])
    temp = []
    for o in env.objs:
        temp.append(o.curr_pos)
    temp = np.vstack(temp)
    x_arr2.append(temp[:,0])
    y_arr2.append(temp[:,1])
    z_arr2.append(temp[:,2])

print('score ', score) 

x_arr = np.vstack(x_arr)
y_arr = np.vstack(y_arr)
z_arr = np.vstack(z_arr)

x_arr2 = np.vstack(x_arr2)
y_arr2 = np.vstack(y_arr2)
z_arr2 = np.vstack(z_arr2)

line, = ax.plot([],[],[], 'bo-', lw=2) # robot at t
line2, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-1
line3, = ax.plot([],[],[], 'bo-',alpha=.3) # robot at t-2 
line4, = ax.plot([],[],[], 'ro', lw=10) # obj at t
line5, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-1
line6, = ax.plot([],[],[], 'ro', lw=10, alpha=.3) # obj at t-2
# line7, = ax.plot([],[],[], 'k-', alpha=.3) # box

j = int(np.round(x_arr.shape[0]))
def update(i):
    global j
    # set robot lines
    thisx = [x_arr[i,0],x_arr[i,1],x_arr[i,2],x_arr[i,3]]
    thisy = [y_arr[i,0],y_arr[i,1],y_arr[i,2],y_arr[i,3]]
    thisz = [z_arr[i,0],z_arr[i,1],z_arr[i,2],z_arr[i,3]]

    line.set_data_3d(thisx,thisy,thisz)
    thisx = [x_arr[i,4],x_arr[i,5],x_arr[i,6],x_arr[i,7]]
    thisy = [y_arr[i,4],y_arr[i,5],y_arr[i,6],y_arr[i,7]]
    thisz = [z_arr[i,4],z_arr[i,5],z_arr[i,6],z_arr[i,7]]
    line2.set_data_3d(thisx,thisy,thisz)
    # n = 3
    # if i > n-1:
    #     lastx = [x_arr[i-n,0],x_arr[i-n,1],x_arr[i-n,2],x_arr[i-n,3]]
    #     lasty = [y_arr[i-n,0],y_arr[i-n,1],y_arr[i-n,2],y_arr[i-n,3]]
    #     lastz = [z_arr[i-n,0],z_arr[i-n,1],z_arr[i-n,2],z_arr[i-n,3]]
    #     line2.set_data_3d(lastx,lasty,lastz)
    # else:
    #     line2.set_data_3d(thisx,thisy,thisz)

    n = 0
    if i > n-1:
        lastx = [x_arr[0,0],x_arr[0,1],x_arr[0,2],x_arr[0,3]]
        lasty = [y_arr[0,0],y_arr[0,1],y_arr[0,2],y_arr[0,3]]
        lastz = [z_arr[0,0],z_arr[0,1],z_arr[0,2],z_arr[0,3]]
        line3.set_data_3d(lastx,lasty,lastz)
    else:
        line3.set_data_3d(thisx,thisy,thisz)

    # set object lines 
    objx,objy,objz = x_arr2[i,:],y_arr2[i,:],z_arr2[i,:]
    line4.set_data_3d(objx,objy,objz)
    n = 3
    if i > n-1:
        lastx = x_arr2[i-n]
        lasty = y_arr2[i-n]
        lastz = z_arr2[i-n]
        line5.set_data_3d(lastx,lasty,lastz)
    else:
        line5.set_data_3d(objx,objy,objz)

    n = 6
    if i > n-1:
        lastx = x_arr2[i-n]
        lasty = y_arr2[i-n]
        lastz = z_arr2[i-n]
        line6.set_data_3d(lastx,lasty,lastz)
    else:
        line6.set_data_3d(objx,objy,objz)

    return line, line2, line3, line4, line5, line6

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d(lims[0,:])
ax.set_xlabel('X')

ax.set_ylim3d(lims[1,:])
ax.set_ylabel('Y')

ax.set_zlim3d(lims[2,:])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
if show_box:
    N = x_box[0,:].shape[0]
    speed = dt*10000/2
else:
    N = x_arr.shape[0]
    speed = dt*1000

ani = animation.FuncAnimation(
    fig, update, N, interval=speed, blit=False)

# ani.save('file.gif')

plt.show()