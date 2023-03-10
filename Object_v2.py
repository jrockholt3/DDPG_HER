from numba import njit, jit
import numba as nb
from numba import int32, float64, float32
from numba.experimental import jitclass
import math as m
import numpy as np

rng = np.random.default_rng()

@njit(float64[:](float64[:],int32,float64[:]),nogil=True)
def rnd_arr(arr, decimal,out):
    return np.round_(arr, decimal, out)

@njit((float64[:])(float64[:],float64,float64[:,:]),nogil=True)
def quantize(arr, res, workspace_limits):
    # this helper function takes in a 3xN array of (x,y,z) coords and
    # outputs the ndx of the coord based on a array representing the whole workspace
    # with resolution: "res"
    range_ = (workspace_limits[:,1] - workspace_limits[:,0])
    ndx_range = range_/res 
    arr = ndx_range * (arr - workspace_limits[:,0]) / range_
    arr = rnd_arr(arr,2,np.zeros_like(arr))
    return arr 

@njit(nogil=True)
def check_range(point, limits):
    if np.all(point >= limits[:,0]) and np.all(point<=limits[:,1]):
        return True 
    else:
        return False

@njit(nogil=True)
def y_solve(x,z,r,pos):
    x_c,y_c,z_c = pos[0],pos[1],pos[2]
    return np.sqrt(np.abs(r**2 - (z-z_c)**2 - (x-x_c)**2))

@njit(nogil=True)
def r_solve(z,r,z_c):       
    return np.sqrt(np.abs(r**2 - (z-z_c)**2))

@njit(nb.types.Tuple((float64[:,:],float64[:,:]))(float64[:],float64,float64,float64[:,:],float64,float64),nogil=True)
def obj_get_coords(curr_pos, t, radius, workspace_limits, res,label):
    x_abs = abs(curr_pos[0]) - radius
    y_abs = abs(curr_pos[1]) - radius
    z_abs = abs(curr_pos[2]) - radius 
    # abs_arr = np.array([x_abs,y_abs,z_abs])
    t_arr = np.array([t])
    if x_abs-radius<=workspace_limits[0,1] and y_abs-radius<=workspace_limits[1,1] and\
            z_abs-radius<=workspace_limits[2,1]: 
        n = int(np.round(2*np.pi**2*radius**2/res**2))
        coord_list = np.zeros((n,4),dtype=float)
        feat_list = np.zeros((n,1),dtype=float)
        ndx = 0
        z = curr_pos[2] - radius
        while np.round(z,2) <= np.round(curr_pos[2] + radius,2):
            # perp distance from x-axis to sphere surface 
            r_slice = r_solve(z,radius,curr_pos[2])
            if np.round(r_slice,2) > 0:
                x = curr_pos[0] - r_slice
                while x <= curr_pos[0] + r_slice:
                    y = curr_pos[1] + y_solve(x,z,radius,curr_pos)
                    point = rnd_arr(np.array([x,y,z]),2,np.zeros(3))
                    if check_range(point, workspace_limits):
                        coord_list[ndx,:] = np.hstack((t_arr, quantize(point,res,workspace_limits)))
                        feat_list[ndx] = label
                        ndx += 1
                    x = x + res # move along x-axis collecting y's
                x = curr_pos[0] + r_slice # reset x
                while x >= curr_pos[0] - r_slice:
                    y = curr_pos[1] - y_solve(x,z,radius,curr_pos)
                    point = rnd_arr(np.array([x,y,z]),2,np.zeros(3))
                    if check_range(point,workspace_limits):
                        coord_list[ndx,:] = np.hstack((t_arr, quantize(point,res,workspace_limits)))
                        feat_list[ndx] = label
                        ndx += 1
                    x = x - res
                z = z + res
            else:
                # the distance from x-axis to the surface is zero (at the top and bottom)
                point = rnd_arr(np.array([curr_pos[0],curr_pos[1],z]),2,np.zeros(3))
                if check_range(point,workspace_limits):
                    coord_list[ndx,:] = np.hstack((t_arr, quantize(point,res,workspace_limits)))
                    feat_list[ndx] = label
                    ndx += 1
                z = z + res
                
    arr1 = coord_list[0:ndx,:]
    arr2 = feat_list[0:ndx]
    return arr1, arr2



class rand_object():
    def __init__(self,object_radius=.03, dt=.01, res=0.01, max_obj_vel = .6, \
                label=1.0,workspace_limits=np.array([[-.6,.6],[-.6,.6],[0,.9]])):
        self.radius = object_radius
        self.res = res
        self.t = 0 # an interger defining the time step 
        self.dt = dt # time between time steps
        self.label = label # classification of the type of object

        # init the object's location at a random (x,y) within the workspace
        self.workspace_limits = workspace_limits
        rho = rng.uniform(.3,.7) * self.workspace_limits[0,1]        
        phi = np.pi/2*rng.random() + 3*np.pi/2
        temp = np.array([-1.0, 1.0])
        phi2 = rng.choice(temp) * np.pi/4 * rng.uniform(0.5, 1.0)
        # setti
        z_max = 0.8*workspace_limits[2,1]
        z_min = 0.1*workspace_limits[2,1]
        self.start = np.array([rho*m.cos(phi), rho*m.sin(phi), rng.uniform(z_min, z_max)])
        self.goal = np.array([rho*m.cos(phi+phi2), rho*m.sin(phi+phi2), rng.uniform(z_min, z_max)])
        v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
        self.vel = (.5*np.random.rand()+.5)*max_obj_vel*v_vec
        self.tf = np.linalg.norm(self.goal - self.start) / np.linalg.norm(self.vel)
        self.original_tf = self.tf
        self.curr_pos = self.start
        if not np.all(np.abs(self.start)-self.radius <= workspace_limits[:,1]):
            print('starting point out of workspace')
        if not np.all(np.abs(self.start)-self.radius <= workspace_limits[:,1]):
            print('end point is out of workspace')

    def set_pos(self, pos):
        self.curr_pos = pos

    def path(self, t, set_new_pos=False):
        goal = self.goal
        start = self.start
        tf = self.tf
        
        
        # letting the object always move around 
        x = t*(goal[0]-start[0])/tf + start[0]
        y = t*(goal[1]-start[1])/tf + start[1]
            
        if set_new_pos == True:
            self.set_pos(np.array([x,y]))
            
        return np.array([x,y])

    def get_coords(self, t):
        coords, feats = obj_get_coords(self.curr_pos, t, self.radius, self.workspace_limits, self.res, self.label)
        return coords.astype(np.int16), feats.astype(np.float32)

    def step(self, time_step=None):
        if time_step == None:
            self.t = self.t + 1
            time_step = self.t 
        else:
            self.t = time_step

        if time_step*self.dt < self.tf:
            self.curr_pos = self.curr_pos + self.vel*self.dt
        else:
            goal = self.goal
            start = self.start
            self.goal = start
            self.start = goal 
            v_vec = (self.goal - self.start) / np.linalg.norm((self.goal - self.start))
            self.vel = np.linalg.norm(self.vel)*v_vec
            self.curr_pos = self.start
            self.tf = self.tf + self.original_tf


    def render(self):
        u = np.linspace(0,2*np.pi,20)
        v = np.linspace(0,np.pi,20)

        x = self.radius * np.outer(np.cos(u),np.sin(v)) + self.curr_pos[0]
        y = self.radius * np.outer(np.sin(u),np.sin(v)) + self.curr_pos[1]
        z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.curr_pos[2]

        return x,y,z



@njit((float64[:])(float64[:,:],float64[:]))
def homo_trans(T, vec):
    """Perform square matrix multiplication of vec = T * vec
    """
    out = np.zeros_like(vec)
    r,c = T.shape
    for i in range(r):
        for j in range(c):
            out[i] += T[i,j] * vec[j]

    return out

@njit(nb.types.Tuple((float64[:,:],float64[:,:]))(float64[:,:],float64,float64[:,:],float64,float64[:,:]))
def cyl_get_coords(original, t, T, res, limits):
    coord_list = np.zeros((original.shape[0],4))
    feat_list = np.zeros((original.shape[0],1))
    t_arr = np.array([t])
    for i in range(original.shape[0]):
        arr_i = homo_trans(T,original[i,:])
        coord_list[i] = np.hstack((t_arr, quantize(arr_i[0:3],res,limits)))
        # coord_list[i] = np.hstack((t_arr, arr_i[0:3]))
        feat_list[i] = 1.0
    return coord_list, feat_list

class Cylinder():
    def __init__(self, r=.02, L=.3, res=.01, label=2, workspace=np.array([[-.6,.6],[-.6,.6],[0,.9]])):
        self.r = r 
        self.L = L
        self.res = res
        self.original = self.make_cloud()
        self.cloud = self.original.copy()
        self.label = label
        self.workspace = workspace

    def make_cloud(self):
        def circle_solve(x,r):
            #return the y-value of x^2+y^2=r^2
            return np.sqrt(np.abs(r**2 - x**2))

        z = 0
        points = []
        while z < self.L:
            x = -self.r 
            while x <= self.r: # positive y values
                y = circle_solve(x, self.r)
                points.append(np.array([x,y,z,1]))
                x += self.res
            while x >= -self.r: # negative y values
                y = -1*circle_solve(x,self.r)
                points.append(np.array([x,y,z,1]))
                x += -1*self.res
            z += self.res
        
        return np.vstack(points)

    def transform(self, T):
        self.cloud = T@self.original

    def plot_cloud(self):
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d
        xx = self.cloud[0,:]
        yy = self.cloud[1,:]
        zz = self.cloud[2,:]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        workspace = self.workspace
        ax.axes.set_xlim3d(left=workspace[0,0], right=workspace[0,1]) 
        ax.axes.set_ylim3d(bottom=workspace[1,0], top=workspace[1,1]) 
        ax.axes.set_zlim3d(bottom=workspace[2,0], top=workspace[2,1]) 
        ax.plot3D(xx,yy,zz)
        plt.show()

    # def down_sample(self,t,T):
    #     dic = dict()
    #     for i in range(1,self.cloud.shape[1]):
    #         arr = np.round(T@self.cloud[:,i])
    #         tup = (arr[0], arr[1], arr[2])
    #         dic[tup] = True

    #     coord_list = []
    #     feat_list = []
    #     for k in dic.keys():
    #         coord_list.append(torch.tensor([t,k[0],k[1],k[2]]))
    #         feat_list.append(torch.tensor(self.label))

    #     return torch.vstack(coord_list), torch.vstack(feat_list)

    def get_coords(self, t, T):
        coords, feat = cyl_get_coords(self.original, t, T, self.res, self.workspace)
        return coords.astype(np.int16), feat.astype(np.float32)