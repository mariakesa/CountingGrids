import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io 

'''The project is unfinished.'''

class CountingGrid(object):
  def __init__(self, size, window_size, nr_of_features):
    '''
    size-- a two-dimensional numpy array, indicating the size of the
    Counting Grid
    window_size-- a two-dimensional numpy array, indicating the size
    of the window
    '''
    self.size = size
    self.window_size = window_size
    self.nr_of_features = nr_of_features # No. of features
    #Initialize arrays here for pi, h
    #pi is a 3D array, storing the probability distributions
    #that collectively characterize the data. The first dimensionality
    #corresponds to features, e.g. it is Z. The second and third
    #dimension correspond to the size of the grid in x and y directions.
    rand_init_pi = 1 + np.random.rand(self.nr_of_features,self.size[0],self.size[1])        
    self.pi = rand_init_pi/sum(rand_init_pi,0)
    #Test pi    
    #self.pi = np.array([[[0.494, 0.524],[0.479,0.418]],[[0.506, 0.476],[0.521, 0.582]]])    
    print self.pi.shape    
    self.h = np.zeros((nr_of_features,self.size[0],self.size[1]))    
    self.compute_histograms()
    
    
  def normalize_data(self,X):
    X=X.transpose()
    normalized_X =  np.prod(self.window_size)*np.divide(X.astype('float'),np.sum(X,0)) 
    normalized_X=normalized_X.transpose()       
    return normalized_X  
    
  def compute_sum_in_a_window(self,grid):
    cumsum1 = grid[self.window_size[0]-1:,self.window_size[1]-1:]
    cumsum2  = grid[:grid.shape[0]-self.window_size[0]+1,self.window_size[1]-1:]
    cumsum3 = grid[self.window_size[0]-1:,:grid.shape[1]-self.window_size[1]+1]    
    cumsum4 = grid[:grid.shape[0]-self.window_size[0]+1,:grid.shape[1]-self.window_size[1]+1]    
    #print cumsum1.shape, cumsum2.shape,cumsum3.shape    
    cumsum =  cumsum1+cumsum2+cumsum3+cumsum4
    cumsum = cumsum[:cumsum.shape[1]-1,:cumsum.shape[1]-1]
    return cumsum
    
    
  def compute_histograms(self):
    '''
    Histograms at each point in the grid are computed
    by averaging the distributions pi in a pre-defined
    window. The left upmost corner of the window is placed
    on the grid position and the distributions are averaged.
    '''
    #Add circular pads to the array, because the grid wraps around itself
    #to remove boundary effects 
    padded_pi=np.lib.pad(self.pi, ((0,0),(0,self.window_size[0]),(0,self.window_size[1])),'wrap')   
    for index_horizontal in range(0,self.pi.shape[1]):
        for index_vertical in range(0,self.pi.shape[2]):
            for feature_index in range(0,self.pi.shape[0]):
                window_array = padded_pi[feature_index,index_horizontal:index_horizontal+self.window_size[0],index_vertical:index_vertical+self.window_size[1]]                       
                self.h[feature_index,index_horizontal,index_vertical]=sum(sum(window_array))/np.prod(self.window_size)
    
  def update_pi(self,X):  
    '''
    Updating the distributions pi on the grid involves
    calculations on data, distributions of mappings of 
    data on the grid log_q and the histograms on each
    grid point.
    '''
    padded_q=np.lib.pad(self.q, ((0,0),(0,self.window_size[0]),(0,self.window_size[1])),'wrap')
    padded_h=np.lib.pad(self.h, ((0,0),(0,self.window_size[0]),(0,self.window_size[1])),'wrap')   
    new_pi=zeros([self.nr_of_features,self.size[0],self.size[1]])     
    for z in range(0,self.nr_of_features):
        for i1 in range(0, self.size[0]):
            for i2 in range(0,self.size[1]):
                t_storage=[]
                for t in range(0,X.shape[0]):
                  interm= X[t,z]*self.compute_sum_in_a_window(np.divide(padded_q[t,:,:],padded_h[z,:,:]))[i1,i2]
                  t_storage.append(t)
                new_pi[z,i1,i2]=self.pi[z,i1,i2]*sum(t_storage)
    self.pi=new_pi
    normalizer=np.sum(self.pi,0)
    self.pi=np.divide(self.pi,normalizer)
    
  def update_h(self):
    self.compute_histograms()
  
  
  #E-step    
  def e_step(self,X):
    '''
    q is a 3D array with shape q.shape=(z_dimension=
    nr_of_samples,x and y=grid_size). 
    It stores the probabilities of a sample mapping to a 
    window in location k=[i1,i2]
    
    h is a 3D array with shape h.shape(z_dimension=
    nr_of_features, x and y=grid_size). 
    h describes the histograms (spanning along the first axis) from 
    which samples are drawn, in each location on the grid k=[i1,i2]
    '''
    nr_of_samples = X.shape[0]
    #Determine a minimal considered probability, for numerical purposes
    min_numeric_probability = float(1)/(10*self.size[0]*self.size[1])
    #Initialize q
    q_size=(nr_of_samples,self.size[0],self.size[1])
    self.q = np.zeros(q_size)
    self.q = np.exp(np.tensordot(X,log(self.h),axes=(1,0)))    
    self.q[self.q<min_numeric_probability]=min_numeric_probability   
    for t in range(0,nr_of_samples):
        normalizer=np.sum(self.q[t,:,:])              
        self.q[t,:,:]= self.q[t,:,:]/normalizer              
    
  #M-step
  def m_step(self,X):
    self.update_pi(X)
    self.update_h()
    
  def fit(self,X,max_iteration,y=None):
    '''
    This is a function for fitting the counting
    grid using variational Expectation Maximization.
    
    The data dimensionality is nr_of_samples on first axis,
    and nr_of_features on second axis.

    X= [nr_of_samples, nr_of_features]    
    '''
    #X=self.normalize_data(X)
    for i in range(0,max_iteration):
      self.e_step(X)
      self.m_step(X)
    
    return self.pi, self.q
    
  def cg_plot(self,labels):
    '''Currently supports 5 different symbols,
    the labels have to be numbers between 0-4
    for the code to work.
    '''
    lab = np.unique(labels)
    L = len(lab)
    for i in range(0,L):
        ids = np.where(labels==lab[i])[0]
        if i==0:
            marker='o'
        if i==1:
            marker='v'
        if i==2:
            marker='^'
        if i==3:
            marker='*'
        if i==4:
            marker='+'
        for t in range(0,len(ids)):
            temp = self.q[ids[t],:,:]
            x,y = np.unravel_index(temp.argmax(), temp.shape)
            noise = 0.2*np.random.rand(1)
            plt.scatter(x+noise,y+noise, marker=marker,s=60,color=cm.rainbow(i*100))
    plt.show()   
    
X=np.array([[1,2],[3,4],[5,6]])
cg_obj=CountingGrid(np.array([3,3]),np.array([2,2]),2)
pi, log_q = cg_obj.fit(X,1)


