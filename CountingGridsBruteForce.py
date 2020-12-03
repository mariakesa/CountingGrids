import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io
import time
import scipy.signal
import scipy

class CountingGridsBruteForce():
    def __init__(self,grid_size,window_size,n_features):
        np.random.seed(7)
        self.grid_size = grid_size
        self.window_size = window_size
        self.n_features = n_features # No. of features
        rand_init_pi = 1 + np.random.rand(self.n_features,self.grid_size[0],self.grid_size[1])
        self.p_grid = rand_init_pi/sum(rand_init_pi,0)
        self.smoothed_grid=np.zeros((self.n_features,self.grid_size[0],self.grid_size[1]))
        self.smooth_grid()


    def smooth_grid(self):
        self.p_grid_pad=np.pad(self.p_grid,pad_width=[(0,0),(self.window_size-1,self.window_size-1),(self.window_size-1,self.window_size-1)],mode='wrap')
        #Naive way to perform convolutions over the
        #window with a average filter.
        start=time.time()
        for z in range(self.n_features):
            for i1 in range(self.grid_size[0]):
                for i2 in range(self.grid_size[0]):
                    #start=time.time()
                    self.smoothed_grid[z,i1,i2]=np.mean(self.p_grid_pad[z,i1:i1+self.window_size,i2:i2+self.window_size]).astype('float32')
                    #end=time.time()
                    #print(end-start)
        end=time.time()
        print(end-start)

start=time.time()
bf=CountingGridsBruteForce([100,100],5,5)
end=time.time()
sm_bf=bf.smoothed_grid
print(sm_bf)

class CountingGrid():
    def __init__(self,grid_size,window_size,n_features):
        np.random.seed(7)
        self.grid_size = grid_size
        self.window_size = window_size
        self.n_features = n_features # No. of features
        rand_init_pi = 1 + np.random.rand(self.n_features,self.grid_size[0],self.grid_size[1])
        self.p_grid = rand_init_pi/sum(rand_init_pi,0)
        self.smoothed_grid=np.zeros((self.n_features,self.grid_size[0],self.grid_size[1]))
        self.smooth_grid()

    def smooth_grid(self):
        start=time.time()
        for z in range(0,self.n_features):
            self.smoothed_grid[z,:,:]=scipy.ndimage.uniform_filter(self.p_grid[z,:,:],size=(self.window_size,self.window_size),mode='wrap').astype('float32')
        end=time.time()
        print(end-start)



print('BREAK')
start_=time.time()
cg=CountingGrid([100,100],5,5)
end_=time.time()
cg_bf=cg.smoothed_grid

print(cg_bf)

print(np.array_equal(sm_bf,cg_bf))
print(end-start)
print(end_-start_)

plt.imshow(sm_bf[0,:,:])
plt.show()
plt.imshow(cg_bf[0,:,:])
plt.show()
