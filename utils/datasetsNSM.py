from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import convolve2d
import skimage.measure
import pandas as pd

import struct
import numpy as np
from numpy import expand_dims
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import sys
#sys.path.append("..") 
#!git clone https://github.com/gussjos/DeepTrack-2.0
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#physical_devices=tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
import scipy.io as IO
import matplotlib.pyplot as plt
from tensorflow import keras
import scipy.io as IO
import os
K=keras.backend
#sys.path.insert(0,'DeepTrack-2.0/')
import deeptrack as dt
# Font parameters
sz = 14
plt.rc('font', size=sz)          # controls default text sizes
plt.rc('axes', titlesize=sz)     # fontsize of the axes title
plt.rc('axes', labelsize=sz)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=sz)    # fontsize of the tick labels
plt.rc('ytick', labelsize=sz)    # fontsize of the tick labels
plt.rc('legend', fontsize=sz)    # legend fontsize
plt.rc('figure', titlesize=sz)  # fontsize of the figure title
import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import tensorflow as tf

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import deeptrack as dt
from deeptrack.features import Feature
import skimage.measure

unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)
print_labels = False

def ConvertTrajToBoundingBoxes(im,length=512,times=128,treshold=0.5):
    debug=False
    if debug:
        plt.close('all')
    # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    

    
    try:            
            nump = im.shape[-1]-2
            batchSize = im.shape[0]
            YOLOLabels = np.zeros((batchSize,nump,5))
            for j in range(0,batchSize):
                for k in range(0,nump):
                    particle_img = im[j,:,:,2+k]
                    particleOccurence = np.where(particle_img>treshold)
                    if np.sum(particleOccurence) <= 0:
                        YOLOLabels = np.delete(YOLOLabels,[j,k],1)
                    else:
                        x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                        y1,y2 = np.min(particleOccurence[0]),np.max(particleOccurence[0])  

                        YOLOLabels[j,k,:] = 0, np.abs(x2+x1)/2/(length-1), (y2+y1)/2/(times-1),(x2-x1)/(length-1),(y2-y1)/(times-1)         

                        if debug:
                            import matplotlib.patches as pch
                            max_nbr_particles = 5
                            nbr_particles = max_nbr_particles
                            plt.figure()#,figsize=(10,2))
                            ax = plt.gca()
                            plt.imshow(particle_img,aspect='auto')
                            ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                            plt.imshow(particle_img,aspect='auto')
                            plt.colorbar()

    except:
           print("Label generation failed. Continuing..")

    

    return YOLOLabels

nump = lambda: np.clip(np.random.randint(5),0,3)
length = 128
L_reduction_factor = 1
reduced_length = int(length/L_reduction_factor)

times = 128
T_reduction_factor = 1
reduced_times = int(times/T_reduction_factor)

# Particle params
Int = lambda : 1e-3*(0.1+0.8*np.random.rand())
Ds = lambda: 0.10*np.sqrt((0.05 + 1*np.random.rand()))
st = lambda: 0.04 + 0.01*np.random.rand()

# Noise params
dX=.00001+.00003*np.random.rand()
dA=0
noise_lev=.0001
biglam=0.6+.4*np.random.rand()
bgnoiseCval=0.03+.02*np.random.rand()
bgnoise=.08+.04*np.random.rand()
bigx0=.1*np.random.randn()

def generate_trajectories(image,Int,Ds,st,nump):
    vel = 0
    length=image.shape[1]
    times=image.shape[0]
    x=np.linspace(-1,1,length)
    t=np.linspace(-1,1,times)
    X, Y=np.meshgrid(t,x)
    f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    
    for p_nbr in range(nump):
        I = Int()
        D = Ds()
        s = st()
        
        # Generate trajectory 
        x0=0
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        
        # Save trajectory with intensity in first image
        image[...,0] *= (1-v1)##(1-v1)

        # Add trajectory to full segmentation image image
        particle_trajectory = np.transpose(f2(1,x0,0.05,0,Y))
        image[...,1] += particle_trajectory 

        # Save single trajectory as additional image
        image[...,-p_nbr-1] = particle_trajectory  
        
    return image

def gen_noise(image,dX,dA,noise_lev,biglam,bgnoiseCval,bgnoise,bigx0):
    length=image.shape[1]
    times=image.shape[0]
    x=np.linspace(-1,1,length)
    t=np.linspace(-1,1,times)
    X, Y=np.meshgrid(t,x)
    f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    bgnoise*=np.random.randn(length)

    tempcorr=3*np.random.rand()
    dAmp=dA#*np.random.rand()
    shiftval=dX*np.random.randn()
    dx=0
    dx2=0
    dAmp0=0
    
    bg0=f2(1,bigx0,biglam,0,x)
    ll=(np.pi-.05)
    
    noise_img = np.zeros_like(image)
    for j in range(times):
        dx=(.7*np.random.randn()+np.sin(ll*j))*dX

        bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
        bgnoiseC/=np.sum(bgnoiseC)
        bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
        dAmp0=dA*np.random.randn()
        bg*=(1+dAmp0)
        noise_img[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
    return noise_img, bg0

def post_process(image,bg0):             
    image[:,:,0]/=bg0 # Normalize image by the bare signal

    image[:,:,0]/=np.mean(image[...,0],axis=0)        
    image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0) # Subtract mean over image

    # Perform same preprocessing as done on experimental images
    ono=np.ones((200,1))
    ono=ono/np.sum(ono)
    image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
    image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")

    image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
    image[:,:,0]*=1000
    
    return image
        
def create_batch(batchsize,times,length,nump):
    nump = nump() # resolve nump for each batch
    batch = np.zeros((batchsize,times,length,nump+2))
    
    for b in range(batchsize):
        image = np.zeros((times,length,nump+2))
        
        # Add noise to image
        noise_image, bg0 = gen_noise(image,dX,dA,noise_lev,biglam,bgnoiseCval,bgnoise,bigx0)
        image = generate_trajectories(noise_image,Int,Ds,st,nump)
        
        # Post process
        image = post_process(image,bg0)
        
        batch[b,...] = image
    
    return batch


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=False, multiscale=False, normalized_labels=True,totalData=1000):
        
        self.img_files = ""

        self.label_files = ""
        
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.totalData = totalData

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        
        batchsize = 1
        im = create_batch(batchsize,times,length,nump)
        print(im.shape)
        v1 = unet.predict(np.expand_dims(im[...,0],axis=-1))
        YOLOLabels = ConvertTrajToBoundingBoxes(im,length=length,times=times,treshold=0.5)
        print(v1.shape)
        print(YOLOLabels.shape)
        

        v1 = np.sum(v1,1).T
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(v1)
        img = torch.cat(3*[img]) # Convert to 3-channel image to simulate RGB information

        # Handle images with less than three channels ## defunct? 
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None       
        boxes = torch.from_numpy(YOLOLabels).reshape(-1,5)#torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        print(img.shape)
        img = np.expand_dims(img,-1)
        return "", img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return self.totalData
