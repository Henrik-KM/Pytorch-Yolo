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


class get_diffusion(Feature):
    
    def __init__(self, vel=0, D=0, I=0, s=0, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        LOW = 0.01
        HIGH = 1.5
        D = (LOW + HIGH*np.random.rand())
        image.append({"D":D})
        
        return image
    
class get_trajectory(Feature):

    
    def __init__(self, vel=0, D=0.1, I=0.1,s=0.05, **kwargs):
        super().__init__(
            vel=vel, D=D, I=I,s=s, **kwargs
        )

    def get(self, image, vel, D, I,s, **kwargs):
        
        D = image.get_property("D")/10
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        G= lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        x0=0
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        image[...,0]*=(1-v1)
        image[...,1]+=np.transpose(f2(1,x0,0.05,0,Y))
        return image
    
class gen_noise(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0,biglam=0,bgnoiseCval=0,bgnoise=0,bigx0=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA,biglam=biglam,bgnoiseCval=bgnoiseCval,bgnoise=bgnoise,bigx0=bigx0, **kwargs
        )

    def get(self, image, noise_lev, dX, dA,biglam,bgnoiseCval,bgnoise,bigx0, **kwargs):
        from scipy.signal import convolve
        
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        G= lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
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
        for j in range(times):
            dx=(.7*np.random.randn()+np.sin(ll*j))*dX
            
            bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
            bgnoiseC/=np.sum(bgnoiseC)
            bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
            dAmp0=dA*np.random.randn()
            bg*=(1+dAmp0)
            image[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
        return image

class post_process(Feature):
    
    
    def __init__(self, noise_lev=0, dX=0, dA=0, **kwargs):
        super().__init__(
            noise_lev=noise_lev, dX=dX, dA=dA, **kwargs
        )

    def get(self, image, **kwargs):
        from scipy.signal import convolve2d
        length=image.shape[1]
        times=image.shape[0]
        x=np.linspace(-1,1,length)
        t=np.linspace(-1,1,times)
        X, Y=np.meshgrid(t,x)
        G= lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
        f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
        
        image[:,:,0]/=np.mean(image[...,0],axis=0)#Normalize image by the bare signal
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)#Subtract mean over image

        #Perform same preprocessing as done on experimental images
        ono=np.ones((200,1))
        ono[0:80]=1
        ono[120:]=1
        ono=ono/np.sum(ono)
        ono2=np.ones((1,50))
        ono2/=np.sum(ono2)
        image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
        image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")
        
        
        
        image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
        a=np.std(image[...,0],axis=0)
        image[:,:,0]/=a
        try:
            image.properties["I"]/=a
        except:
            pass
        
        return image

    
class input_array(Feature):
    __distributed__ = False
    def get(self,image, **kwargs):
        image=np.zeros((times,length,2))
        return image

        
def batch_function(image):
    img = image[...,:1]
    img = skimage.measure.block_reduce(img,(T_reduction_factor,L_reduction_factor,1),np.mean)
    GAN_output = unet.predict(np.expand_dims(img,axis=0))
    #mask = skimage.measure.block_reduce(GAN_output, (1,64,16,1), np.mean)
    #img = np.expand_dims(img,axis=0)
    #print('GAN_output[0].shape',GAN_output[0].shape)
    return GAN_output[0] #, mask

def label_function(image):

    
    t_final = 8
    L_final = 8
    time_reduction = int(image.shape[0]/t_final)
    len_reduction = int(image.shape[1]/L_final)
    
    mask = skimage.measure.block_reduce(image[...,1:2], (time_reduction,len_reduction,1), np.mean)
    
    #print('mask.shape',mask.shape)

    labels = np.ones((t_final,L_final,2))
    labels[...,0] *= D
    labels[...,1:] = mask
    
    return labels



        
def ConvertTrajToBoundingBoxes(v1,batchSize,nump,length=512,times=128,treshold=0.5):
    YOLOLabels = np.zeros((batchSize,nump,5)) # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    try:
        for j in range(0,batchSize):
            for k in range(0,nump):
                p = v1[j,k,...]
                particleOccurence = np.where(p>treshold)
                if not np.size(particleOccurence)  == 0:
                    x1,x2 = particleOccurence[0][0],particleOccurence[0][-1]
                    y1,y2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])
                    
                    YOLOLabels[j,k,:] = 0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)
                else:
                    YOLOLabels = np.delete(YOLOLabels,[j,k],1)
    except:
        print("Label generation failed. Continuing..")

    return YOLOLabels


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
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.totalData = totalData
        unet = tf.keras.models.load_model('../input/network-weights/unet-1-dec-1415.h5',compile=False)

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        print_labels = False
        
        batchSize = 1
        Int = lambda: 1.05e-3*(0.08+0.8*np.random.rand())
        Ds = lambda: 0.10*np.sqrt((0.05 + 1.0*np.random.rand()))
        st = lambda: 0.04 + 0.01*np.random.rand()
        
    
        length = 512
        times = 128
        
        dA = lambda: 0.00006 * (0.7 + np.random.rand())
        dX = lambda: 0.00006* (0.7 + np.random.rand())
        bgnoiselev = lambda: 0.0006* (0.7 + np.random.rand())
    
        time_reduction = 16
        length_reduction = 64
        downsampling_factor_for_length = 1
        nump=2 #number of particles
        
        batch_size = 16
        min_data_size = 64
        max_data_size = 128
        
        image=dt.FlipLR(dt.FlipUD(input_array() + get_diffusion() + #set_params(D=Ds, I=Int, s=st) + #get_diffusion(D=Ds) + get_intensity(I=Int,s=st) + 
                          gen_noise(dX=.00001+.00003*np.random.rand(),
                                                  dA=0,
                                                  noise_lev=.0001,
                                                  #noise_lev = 0.0001*(0.75+0.75*np.random.rand()),
                                                  biglam=0.6+.4*np.random.rand(),
                                                  bgnoiseCval=0.03+.02*np.random.rand(),
                                                  #bgnoiseCval=0.03+.05*np.random.rand(),
                                                  #bgnoise=.08+.14*np.random.rand(),
                                                  bgnoise=.08+.04*np.random.rand(),
                                                  bigx0=lambda: .1*np.random.randn())
                                      + get_trajectory(I=Int,s=st)**nbrParticles
                                      +post_process()))
        
        data_generator = Generate8b8batchboxesv2Generator(bgnoiselev=bgnoiselev,
                                                         Int=Int,
                                                         st=st,
                                                         Ds=Ds, 
                                                         dA=dA,
                                                         dX=dX,
                                                         batchsize=batchSize,
                                                         length=length,
                                                         times=times,
                                                         bgexp=None,
                                                         print_labels=print_labels,
                                                         time_reduction=time_reduction,
                                                         length_reduction=length_reduction,
                                                         downsampling_factor_for_length=downsampling_factor_for_length,
                                                         nump=nump)
        val,valL,v1 = next(data_generator)
        
        data_generator = dt.generators.ContinuousGenerator(image,
                                    batch_function=batch_function,
                                    label_function=label_function,
                                    batch_size=batch_size,
                                    min_data_size=min_data_size,
                                    max_data_size=max_data_size) 
        
        # For training on iOC = 5e-4, D = [10,20,50] mu m^2/s
        # Range on Ds: 0.03 -> 0.08
        # Range on Is: 5e-3 = good contrast
        
        # Plot predictions of validation samples
        val=val[...,:1]
        valld = valL[0]
        
        YOLOLabels=ConvertTrajToBoundingBoxes(v1,batchSize,nump,length=512,times=128,treshold=0.5)
       # v1 = np.sum(v1[0,...],0).T #Place all particles in the same image
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
