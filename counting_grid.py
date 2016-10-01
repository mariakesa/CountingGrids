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
    self.alpha=1e-10
    #Initialize arrays here for pi, h
    #pi is a 3D array, storing the probability distributions
    #that collectively characterize the data. The first dimensionality
    #corresponds to features, e.g. it is Z. The second and third
    #dimension correspond to the size of the grid in x and y directions.
    rand_init_pi = 1 + np.random.rand(self.nr_of_features,self.size[0],self.size[1])        
    self.pi = rand_init_pi/sum(rand_init_pi,0) 
    print self.pi
    #self.pi = np.array([[[0.494, 0.524],[0.479,0.418]],[[0.506, 0.476],[0.521, 0.582]]])          
    self.h = np.zeros((nr_of_features,self.size[0],self.size[1]))    
    self.compute_histograms()
    print 'Beginning',self.pi[0,0,0]
    
    
  def normalize_data(self,X):
    X=np.exp(X/100)
    X=X.transpose()
    normalized_X =  100*np.prod(self.window_size)*np.divide(X.astype('float'),np.sum(X,0)) 
    normalized_X=normalized_X.transpose()       
    return normalized_X  
    
  def compute_sums_over_windows(self, array):
    '''
    Function for efficiently computing sums over windows of the array for the 
    averaged histograms.
    '''
    xW=self.window_size[0]
    yW=self.window_size[1]
    first_term=array[:,xW:,yW:]
    second_term=array[:,0:array.shape[1]-xW,yW:]
    third_term=array[:,xW:,0:array.shape[1]-yW]
    fourth_term=array[:,0:array.shape[1]-xW,0:array.shape[1]-yW]
    sums=first_term-second_term-third_term+fourth_term
    return sums
  
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
    #Compute cumsums and pad them to pass on for computing window sums over pi
    cumsums=np.lib.pad(np.cumsum(np.cumsum(padded_pi,axis=1), axis=2),((0,0),(1,0),(1,0)), mode='constant',constant_values=(0,0))    
    unnormalized_h=self.compute_sums_over_windows(cumsums)[:,0:self.h.shape[1],0:self.h.shape[2]]
    self.h=unnormalized_h
    normalizer=np.sum(self.h,0)
    self.h=np.divide(self.h,normalizer)
    
  def update_pi(self,X):  
    '''
    Updating the distributions pi on the grid involves
    calculations on data, distributions of mappings of 
    data on the grid, q and the histograms on each
    grid point.
    '''
    padded_q=np.lib.pad(self.q, ((0,0),(0,self.window_size[0]),(0,self.window_size[1])),'wrap')   
    padded_h=np.lib.pad(self.h, ((0,0),(0,self.window_size[0]),(0,self.window_size[1])),'wrap') 
    #Add small additive factor for numerical reasons    
    padded_h+=self.alpha*np.prod(self.window_size)
    scalar_prod_X_q=np.dot(X.T,np.reshape(padded_q, [self.q.shape[0],np.prod(self.size+self.window_size)]))
    scalar_prod_X_q=np.reshape(scalar_prod_X_q,[self.nr_of_features,self.size[0]+self.window_size[0],self.size[1]+self.window_size[1]])   
    division=np.divide(scalar_prod_X_q,padded_h)
    #Use cumsums to prepare for computation of sums over windows
    cumsums=np.cumsum(np.cumsum(division,axis=1),axis=2)
    combined_X_q_h=self.compute_sums_over_windows(cumsums)
    
    #Filter out entries which 
    #mask= np.sum(unnormalized_pi, axis=0)!=0
    #print mask
    #print 'ziu',np.multiply(self.pi,mask)
    '''
    for z in range(0,self.nr_of_features):
        sample_array=np.zeros(self.size[0],self.size[1])
        for t in range(0,X.shape[0]):
            interm= X[t,z]*self.compute_sum_in_a_window(np.divide(padded_q[t,:,:],padded_h[z,:,:]))
            j=self.compute_sum_in_a_window(np.divide(padded_q[t,:,:],padded_h[z,:,:])) 
            sample_array=sample_array+interm
        self.pi[z,:,:]=np.multiply(sample_array,self.pi[z,:,:])   
    normalizer=np.sum(self.pi,0)
    self.pi=np.divide(self.pi,normalizer)
    '''
    
    
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
    #Alternative way to compute log_q
    #log_q = np.tensordot(X,np.log(self.h),axes=(1,0))
    q_size=(nr_of_samples,self.size[0],self.size[1])
    self.q = np.zeros(q_size)    
    #The aim of this step is to take dot products over the features in the 
    #data array and their probabilities in the averaged histograms h_k,z.
    #This is achieved by first reshaping the h array.           
    log_q=np.dot(X,np.reshape(np.log(self.h),[self.nr_of_features,self.size[0]*self.size[1]]))
    #Normalization in the log-domain with an exp-normalization trick
    #See http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ 
    scaled=log_q.T-np.amax(log_q,axis=1)
    log_q= scaled-logsumexp(scaled,axis=0)
    log_q=log_q.T
    log_q=np.reshape(log_q,[nr_of_samples,self.size[0],self.size[1]])  
    self.q=np.exp(log_q)
    #Filter out tiny probabilities for numerical reasons
    min_numeric_probability = float(1)/(10*self.size[0]*self.size[1])    
    self.q[self.q<min_numeric_probability]=min_numeric_probability     
    #Normalize array    
    normalizer=np.sum(np.sum(self.q,axis=1),axis=1)
    self.q=(self.q.T/normalizer).T
               
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
    X=self.normalize_data(X)
    for i in range(0,max_iteration):
      print 'iteration', i
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
    
X=np.array([[1,2],[100,200],[5000,6]])
cg_obj=CountingGrid(np.array([3,3]),np.array([2,2]),2)
pi, log_q = cg_obj.fit(X,1)

'''
h=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
reshape=np.reshape(h,[2,9])
print reshape
product=np.dot(X,reshape)
print product
reshape2=np.reshape(product,[3,3,3])
print reshape2
'''
h=np.array([[1,2,3],[4,5,6]])
z=np.reshape(h,(3,2),order='F')
print z
