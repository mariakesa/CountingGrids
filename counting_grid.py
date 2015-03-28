import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
This is an implementation of the CountingGrid model. Right now it has a bug that
I am working to find and fix. So don't use it yet!
Reference: N.Jojic and A.Perina. "Multidimensional counting grids: Inferring word order from disordered bags of words."
In Conference on Uncertainty in Artificial Intelligence, UAI 2011
'''

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
    self.compute_histograms()
    
  def normalize_data(self,X):
    X=X.transpose()
    normalized_X =  np.prod(self.window_size)*np.divide(X.astype('float'),np.sum(X,0)) 
    normalized_X=normalized_X.transpose()       
    return normalized_X  
    
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
    #The following code computes histograms as averages of distributions pi 
    #in a sliding window across the grid
    interm_h=np.cumsum(padded_pi,1)
    interm_h=np.transpose(interm_h,(0,2,1)) 
    interm_h=np.cumsum(interm_h,1)
    interm_h=np.transpose(interm_h,(0,2,1))  
    interm_h=np.lib.pad(interm_h,((0,0),(1,0),(1,0)),'constant',constant_values=(0,0))        
    self.h=self.averaging_procedure(interm_h,True)

    
  def averaging_procedure(self,intermediate,normalize):
    h1=intermediate[:,self.window_size[0]:,self.window_size[1]:]   
    h2=intermediate[:,0:intermediate.shape[1]-self.window_size[0],self.window_size[1]:]      
    h3=intermediate[:,self.window_size[0]:,:intermediate.shape[2]-self.window_size[1]]    
    h4=intermediate[:,0:intermediate.shape[1]-self.window_size[0],0:intermediate.shape[2]-self.window_size[1]]       
    h= h1-h2-h3+h4     
    if normalize:
        h= h[:,0:-1,0:-1]     
        normalizer=np.sum(h,0)      
        h=np.divide(h.astype(float),normalizer)
    return h
    
  def update_pi(self,X):  
    '''
    Updating the distributions pi on the grid involves
    calculations on data, distributions of mappings of 
    data on the grid log_q and the histograms on each
    grid point.
    '''
    alpha = 1e-10
    #Compute the term corresponding to multiplication and summation of counts and
    #the mapping of samples on the grid
    padded_q = np.lib.pad(np.exp(self.log_q), ((0,0),(self.window_size[0],0),(self.window_size[1],0)),'wrap') 
    nr_of_samples=self.log_q.shape[0]
    padded_q = np.reshape(padded_q,(nr_of_samples,np.prod(self.size+self.window_size)),'F')
    padded_q=np.dot(padded_q.transpose(),X)
    padded_q=np.reshape(padded_q.transpose(),(self.nr_of_features,self.size[0]+self.window_size[0],self.size[1]+self.window_size[1]),'F')       
    #Okay, up to here
    #Add small number to h for numerical reasons
    h = self.h+self.window_size[0]*self.window_size[1]*alpha
    padded_h = np.lib.pad(h,((0,0),(self.window_size[0],0),(self.window_size[1],0)),'wrap')           
    #okay upto here
    composite_q_h=np.divide(padded_q,padded_h)          
    #Okay up to here
    composite_q_h=np.transpose(np.cumsum(np.transpose(np.cumsum(composite_q_h,1),(0,2,1)),1),(0,2,1))
    #Okay up to here 
    composite_q_h=self.averaging_procedure(composite_q_h,False)   
    #Okay up to here    
    #Remove negative values    
    replacement_indices = np.where(composite_q_h< 0)    
    composite_q_h[replacement_indices]=0 
    #okay up to here
    pseudocounts= np.mean(np.sum(X,0).astype('float')/np.prod(self.size))/2.5
    #Okay up to here    
    un_pi=pseudocounts+np.multiply(composite_q_h,self.pi+alpha)    
    mask = (sum(un_pi,0)!=0).astype(float)
    interm_pi = np.divide(un_pi,sum(un_pi,0))
    interm_pi = np.multiply(interm_pi,mask)
    correction= np.multiply((float(1)/self.nr_of_features)*np.ones((self.nr_of_features,self.size[0],self.size[1])),np.logical_not(mask).astype('float'))
    self.pi=interm_pi+correction
  
  def update_h(self):
    self.compute_histograms()
  
  
  #E-step    
  def e_step(self,X):
    '''
    log_q is a 3D array with shape q.shape=(z_dimension=
    nr_of_samples,x and y=grid_size). 
    It stores the log probabilities of a sample mapping to a 
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
    log_q_size=(nr_of_samples,self.size[0],self.size[1])
    self.log_q = np.zeros(log_q_size)
    interm_q = np.reshape(np.log(self.h),(self.nr_of_features,self.size[0]*self.size[1]),'F') 
    interm_q = np.dot(interm_q.transpose(),X.transpose())        
    #Replace values that fall below a threshold for numerical stability
    numerical_manipulation=np.subtract(interm_q-np.amax(interm_q,0),logsumexp(interm_q-np.amax(interm_q,0),0)) 
    numerical_manipulation = np.reshape(numerical_manipulation.transpose(),(nr_of_samples,self.size[0],self.size[1]),'F')
    numerical_manipulation = np.exp(numerical_manipulation)        
    replacement_indices = np.where(numerical_manipulation< min_numeric_probability)  
    numerical_manipulation[replacement_indices]=min_numeric_probability
    #Normalize the probability distributions    
    for t in range(0,nr_of_samples):
        normalizer=np.sum(numerical_manipulation[t,:,:])              
        self.log_q[t,:,:]= np.log(numerical_manipulation[t,:,:]/normalizer)              
    
    
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
      self.e_step(X)     
      self.m_step(X)
    
    return self.pi, self.log_q
    
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
            temp = self.log_q[ids[t],:,:]
            x,y = np.unravel_index(temp.argmax(), temp.shape)
            noise = 0.2*np.random.rand(1)
            plt.scatter(x+noise,y+noise, marker=marker,s=60,color=cm.rainbow(i*100))
    plt.show()       

