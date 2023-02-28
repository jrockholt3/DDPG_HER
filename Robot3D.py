#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import torch 
import numpy as np
import math as m
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from Object_v2 import Cylinder

# Goal is to create a joint space that the robot can operate in. As long as a decision doesn't put it out of this joint space it can go there. This could be constrain by the orientation of the part.   

# In[2]:


# settings
links = np.array([0,.3,.3])
workspace = np.sum(links)
max_obj_vel = workspace # takes 3 seconds to move across the entire workspace
workspace_limits = np.array([[-workspace, workspace],[-workspace, workspace],[0, .9]])
res = 0.01
# dt = .01


# In[3]:


def c(x):
    return m.cos(x)
def s(x):
    return m.sin(x)

def T_ji(thj, aij, lij, Sj):
    return np.array([[       c(thj),        -s(thj),        0,        lij],
                     [s(thj)*c(aij),  c(thj)*c(aij),  -s(aij),  -s(aij)*Sj],
                     [s(thj)*s(aij),  c(thj)*s(aij),   c(aij),   c(aij)*Sj], 
                     [      0,             0,             0,             1]])

def T_inverse(T):
    R = T[0:3,0:3].T
#     print(R)
    shift = T[0:3,3]
#     print('shift ' + str(shift))
    shift = -R@shift
#     print('-shift ' + str(shift))
    new_T = np.zeros((4,4))
    for i in range(0,3):
        for j in range(0,3):
            new_T[i,j] = R[i,j]
    for i in range(0,3):
        new_T[i,3] = shift[i]
    new_T[3,3] = 1
    
#     print(new_T)
    return new_T

def T_ij(thj, aij, lij, Sj):
    T = T_ji(thj, aij, lij, Sj).T
    shift = np.array([-c(thj)*lij, s(thj)*lij, -Sj, 1])
    T[3,0:2] = 0
    T[:,3] = shift[:]
    return T

def T_1F(ph, S):
    return np.array([[c(ph), -s(ph), 0, 0],
                     [s(ph),  c(ph), 0, 0],
                     [    0,      0, 1, S], 
                     [    0,      0, 0, 1]])


    
# In[4]:

class robot_3link():
    def __init__(self, label=np.array([1,0,0])):
        self.base = np.array([0,0,0,1])
        self.links = np.array([0,.3,.3])
        self.aph = np.array([np.pi/2, 0.0, 0.0]) # twist angles
        self.pos = np.array([0.0, 0.0, 0.0]) # joint angles
        self.S = np.array([0.3, 0.0, 0.0]) # joint offset
        self.P_3 = np.array([self.links[2],0,0,1]) # tool point as seen by 3rd coord sys
        self.v_lim = np.array([m.pi, m.pi, m.pi]) # joint velocity limits 
        self.jnt_vel = np.array([0.0, 0.0, 0.0])
        self.traj = np.array([])
        self.label = label
        self.body = Cylinder()
        self.jnt_max = np.array([np.inf, .9*np.pi, .9*np.pi])
        self.jnt_min = np.array([-np.inf, -.9*np.pi, -.9*np.pi])

    def set_pose(self, th_arr):
        c = np.cos(th_arr)
        s = np.sin(th_arr)
        self.pos = np.arctan2(s,c)

    def get_pose(self):
        return self.pos
    
    def set_jnt_vel(self, vel_arr):
        self.jnt_vel = vel_arr
        
    def asb_bodyframe(self, obj_pos):
        T_dict = self.get_transform(self.pos)
        keys = T_dict.keys()
        vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
        temp = np.ones((4,3))
        for i,k in enumerate(keys):
            T = T_dict[k]
            T = T_inverse(T)
            vec_T = T@vec # pos of object in frame pov
            temp[:,i] = vec_T # gives the relative position of the vector
        
        # w = self.jnt_vel
        # th = self.pos
        # vp_f = np.array([scene_obj.vel[0], scene_obj.vel[1],0])
        
        return temp
    
    def proximity(self, scene_obj_loc):
        obj_asb_bodies = self.asb_bodyframe(scene_obj_loc)
        prox_arr = np.zeros((len(self.links)))
        # body 1
        for i in range(0,len(self.links)):
            obj_pos = obj_asb_bodies[0:3,i]
            if obj_pos[0] <= 0: # if obj is behind ith joint
                prox1 = np.linalg.norm(obj_pos)
            elif obj_pos[0] >= self.links[i]: # if obj is past the jth joint
                prox1 = np.linalg.norm(obj_pos - np.array([self.links[i],0,0]))
            else:
                prox1 = np.linalg.norm(obj_pos[1:]) # norm distance from arm
                
            prox_arr[i] = prox1
            
        return prox_arr
    
    def forward(self, th=None, make_plot=False, with_obj=False, obj=None):
        if np.any(th==None): th = self.pos
        th = th + np.array([0,np.pi/2,0])
        l = self.links
        a = self.aph
        S = self.S 
        temp = np.vstack((self.base, 
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@np.array([0,0,0,1]),
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@np.array([0,0,0,1]),
                         T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@self.P_3))
        if make_plot == True:
            if not with_obj:
                self.plot_pose(arr=temp)
            else:
                self.plot_pose(arr=temp,with_obj=True,obj=obj)
        
        return temp.T
    
    def get_transform(self, th=None):
        if np.any(th==None): th = self.pos
        th = th + np.array([0,np.pi/2,0])
        l = self.links
        a = self.aph
        S = self.S
        dict = {'1toF':T_1F(th[0],self.S[0]), # 1st body
                '2toF':T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1]), # second body 
                '3toF':T_1F(th[0],self.S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])} # third body
        return dict
     
    def reverse(self, goal, make_plot=False):
        x = goal[0]
        y = goal[1]
        z = goal[2]
        
        # th1 is the angle from x-axis of a12 in the positive RH sense. two solutions
        th1 = np.array([m.atan2(y,x), m.atan2(y,x)])
        s1 = np.sin(th1)
        c1 = np.cos(th1)
        th1 = np.arctan2(s1,c1)

        r = np.sqrt(x**2 + y**2)
        z = z - self.S[0]
        alpha = np.arctan2(z,r)
        
        c = self.links[1]
        a = self.links[2]
        b = np.sqrt(r**2 + z**2)
        
        A = np.arccos((a**2 - b**2 - c**2) / (-2*b*c)) # cosine law
        if np.isnan(A).any():
            assert('got an nan')
        
        th2 = np.array([A + alpha, alpha - A])
        
        val = (b**2 - a**2 - c**2) / (-2*a*c)
        if abs(np.round(val,5)) == 1:
            th3 = np.array([0,0])
        else:
            B = np.arccos(val)
            th3 = np.array([B-np.pi, np.pi-B])

        # shift the middle joint by pi/2 (zero when stick straight up)
        th = np.vstack((th1, th2-np.pi/2, th3))
        if make_plot == True: 
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            sol1 = self.forward(th[:,0])
            xx = sol1[0,:]
            yy = sol1[1,:]
            zz = sol1[2,:]
            ax.plot3D(xx,yy,zz,'red', label='Sol1')
            ax.scatter3D(xx,yy,zz,'r')
            sol1 = self.forward(th[:,1])
            xx = sol1[0,:]
            yy = sol1[1,:]
            zz = sol1[2,:]
            ax.plot3D(xx,yy,zz,'blue', label='Sol2')
            ax.scatter3D(xx,yy,zz,'b')
            ax.axes.set_xlim3d(left=-workspace, right=workspace) 
            ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
            ax.axes.set_zlim3d(bottom=0, top=workspace+self.S[0])
            ax.legend()
            plt.show()
            
        return th
    
    def plot_pose(self, th=None, arr = None, with_obj=False, obj=None):
        if np.any(th==None) and np.any(arr==None): 
            th = self.pos
            arr = self.forward(th=th)
        elif np.any(arr==None) and np.all(th!=None):
            arr = self.forward(th=th)
            
        xx = arr[:,0]
        yy = arr[:,1]
        zz = arr[:,2]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xx,yy,zz)
        ax.plot3D(xx,yy,zz,'red')
        ax.axes.set_xlim3d(left=-workspace, right=workspace) 
        ax.axes.set_ylim3d(bottom=-workspace, top=workspace) 
        ax.axes.set_zlim3d(bottom=0, top=workspace+self.S[0]) 
        if with_obj:
            for o in obj:
                x,y,z = o.curr_pos[0],o.curr_pos[1],o.curr_pos[2]
                ax.scatter3D(x,y,z)
        plt.show()

    def get_coords(self,t):
        T_dict = self.get_transform()
        T2 = T_dict['2toF']
        T3 = T_dict['3toF']
        th = np.pi/2
        Ty = np.array([[np.cos(th), 0.0, np.sin(th), 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [-np.sin(th), 0.0, np.cos(th), 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

        T = T2@Ty
        original = self.body.original
        coords2,feat2 = self.body.get_coords(t, T)
        T = T3@Ty
        coords3,feat3 = self.body.get_coords(t, T)

        return np.vstack((coords2,coords3)), np.vstack((feat2, feat3))

    def plot_cloud(self):
        coords,_ = self.get_coords(0)
        # t = coords[:,0]
        # coords = coords[t==0]
        xx = coords[:,1]
        yy = coords[:,2]
        zz = coords[:,3]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xx,yy,zz)
        ax.set_xlim3d(left=0, right=120)
        ax.set_ylim3d(bottom=0, top=120)
        ax.set_zlim3d(bottom=0, top=90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def check_safety(self):
        min_r = .05 # distance that robot can be to itself
        violation = False
        pos = self.pos
        temp = self.forward()
        temp = temp[:,3]
        prox = self.proximity(temp)
        prox = prox[0]
        # check for joint limit violations
        if np.any(pos > self.jnt_max) or np.any(pos < self.jnt_min):
            # print('joint limits')
            violation = True 
            vel = self.jnt_vel
            # all joints greater than lim, set vel to zero
            vel[pos >= self.jnt_max] = 0 
            vel[pos <= self.jnt_min] = 0 
            self.set_jnt_vel(vel)
            # return joint to equal limit
            pos = np.clip(pos,self.jnt_min,self.jnt_max)
            self.set_pose(pos)

        # check for collison with base
        if prox < min_r:
            # print('collison')
            violation = True 
            xy = np.array([temp[0],temp[1]])
            vec = xy/np.linalg.norm(xy)
            new_xy = min_r * vec # reset eef point r_min away from base
            th_arr = self.reverse(np.array([new_xy[0],new_xy[1],temp[2]]))
            dth1 = np.linalg.norm(th_arr[:,0] - self.pos)
            dth2 = np.linalg.norm(th_arr[:,1] - self.pos)
            if dth1 < dth2:
                self.set_pose(th_arr[:,0])
            else:
                self.set_pose(th_arr[:,1])
            self.set_pose(th_arr[:,0])
            # set top joints vel to zero bc eef just hit base joint
            vel = np.array([self.jnt_vel[0],0,0])
            self.set_jnt_vel(vel)
        
        # check if eef is going through the floor
        if temp[2] <= min_r:
            # print('hitting floor')
            violation = True
            # temp[2] = min_r
            th_arr = self.reverse(temp)
            dth1 = np.linalg.norm(th_arr[:,0] - self.pos)
            dth2 = np.linalg.norm(th_arr[:,1] - self.pos)
            if dth1 < dth2:
                ndx = 0
            else:
                ndx = 1
            temp[2] = min_r
            th = self.reverse(temp)
            th = th[:,ndx]
            self.set_pose(th)
            vel = np.array([self.jnt_vel[0],0,0])
            self.set_jnt_vel(vel)

        return violation





