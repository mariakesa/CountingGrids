import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io

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
        for z in range(self.n_features):
            for i1 in range(self.grid_size[0]):
                for i2 in range(self.grid_size[0]):
                    print(i1,i2)
                    self.smoothed_grid[z,i1,i2]=np.mean(self.p_grid_pad[z,i1:i1+self.window_size,i2:i2+self.window_size])
                    print(self.smoothed_grid[z,i1,i2])
        print(self.smoothed_grid)
        print(self.smoothed_grid.shape)

CountingGridsBruteForce([10,10],4,5)
