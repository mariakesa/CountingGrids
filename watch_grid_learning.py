from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets


import time
import numpy as np
from vispy import app
import vispy

from PyQt5.QtWidgets import *
import vispy.app
import sys

from vispy.app import use_app
use_app('PyQt5')
from vispy import scene
from vispy import color
from vispy.color.colormap import Colormap

import h5py

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io
import time
import scipy.signal
import scipy


import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io
import time
import scipy.signal
import scipy
import torch

from scipy import signal



import imageio
from vispy import visuals

from skimage.transform import rescale, resize, downscale_local_mean

class Canvas(scene.SceneCanvas):

    def __init__(self):
        scene.SceneCanvas.__init__(self,keys='interactive', size=(600, 600))

        self.unfreeze()

        self.load_p_grid()
        print(self.p_grid.shape)

        self.i=20

        self.t=0

        self.view=self.central_widget.add_view()
        from skimage.transform import rescale, resize, downscale_local_mean
        image_resized = resize(self.p_grid[self.t,self.i,:,:], (30* 20, 30*20),
                       anti_aliasing=True)
        self.scaler=100
        print(image_resized*10000)
        from skimage.transform import rescale, resize, downscale_local_mean
        self.image=scene.visuals.Image(self.scaler*image_resized,parent=self.view.scene, cmap='bwr',clim=[0,1])
        print('success')

    def load_p_grid(self):
        self.p_grid=np.load('p_grid.npy')



class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        self.pl_ind_box=QLineEdit()
        self.pl_ind_box.setText("0")
        self.pl_ind_box.setFixedWidth(35)
        self.l0.addWidget(self.pl_ind_box, 0, 4, 1, 2)
        self.pl_ind_box.returnPressed.connect(lambda: self.change_plane_ind())
        widget.setLayout(self.l0)

        self.timer_init()


    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=2
        canvas.t=0



    def update(self,ev):
        canvas.t+=1
        print(canvas.t)
        image_resized = resize(canvas.p_grid[canvas.t,canvas.i,:,:], (30* 20, 30*20),
                       anti_aliasing=True)
        canvas.image.set_data(canvas.scaler*image_resized)
        if canvas.t>=10:
            canvas.t=0
        canvas.update()


    def change_plane_ind(self):
        canvas.i=int(self.pl_ind_box.text())
        self.timer_init()

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
X = filter_by_variance(X,200)
#Compose labels matrix from filec
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
X=X/np.sum(X,axis=0)

canvas = Canvas()
vispy.use('PyQt5')
w = MainWindow(canvas)
w.show()
vispy.app.run()
