from scipy import io 
import numpy as np
from counting_grid import CountingGrid

'''
This is an example of using the CountingGrid
class on lung cancer gene expression data.
'''

def filter_by_variance(X,nr_of_features_to_keep):
    '''
    Function for thresholding data by variance,
    keeping only 'nr_of_features_to_keep' features
    with the highest variance.
    X=[nr_of_samples, nr_of_features]
    '''    
    ordering = np.argsort(np.var(X,axis=0))[::-1]
    threshold = ordering[0:nr_of_features_to_keep]
    X=X[:,threshold]
    return X

#Preparing the data
data = io.loadmat('lung_bhattacherjee.mat')
X= data['data']
Y_labels = data['sample_names'][0]
X= X.T
X = filter_by_variance(X,500)
#Compose labels matrix from file
Y=np.zeros((len(Y_labels),1))
for j in range(0,len(Y_labels)):
    if str(Y_labels[j][0])[0:3]=='AD-':
        Y[j]=0
    if str(Y_labels[j][0])[0:3]=='NL-':
        Y[j]=1
    if str(Y_labels[j][0])[0:3]=='SMC':
        Y[j]=2
    if str(Y_labels[j][0])[0:3]=='SQ-':
        Y[j]=3
    if str(Y_labels[j][0])[0:3]=='COI':
        Y[j]=4

#Usage
cg_obj=CountingGrid(np.array([15,15]),np.array([3,3]),500)
pi, log_q = cg_obj.fit(X,100)
cg_obj.cg_plot(Y)
