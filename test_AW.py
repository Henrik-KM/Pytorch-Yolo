from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import multiprocessing

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def main():
    batch_size=8
    model_def = "config/yolov3-custom.cfg"
    data_config = "config/custom.data"
    weights_path = "weights/yolov3.weights"
    class_path = "data/custom/classes.names"
    iou_thres = 0.5
    conf_thres = 0.001
    nms_thres = 0.5
    n_cpu = 8
    img_size = 416

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device', device)

    data_config = parse_data_config(data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    
    # Initiate model
    model = Darknet(model_def, img_size=img_size).to(device)
    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))
    
    print("Compute mAP...")
    
    # precision, recall, AP, f1, ap_class = evaluate(
    #     model,
    #     path=valid_path,
    #     iou_thres=iou_thres,
    #     conf_thres=conf_thres,
    #     nms_thres=nms_thres,
    #     img_size=img_size,
    #     batch_size=8,
    # )
    path=valid_path

    list_path=path
    imgFiles = []
    with open(list_path, "r") as file:
        basePath = "data/custom/images/"#file.readlines()[0]
        for animal in os.listdir(basePath):
            for img in os.listdir(basePath+"/"+animal):
                imgFiles = np.append(imgFiles,basePath+"/"+animal+"/"+img)
                with open("valid.txt", "a+") as file:
                    file.write(basePath+animal+"/"+img+"\n")

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn
    )
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
   
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        #print(batch_i)
        #print(imgs)
        #print(targets)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        
    
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
    
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
    
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        #print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        print("Class ", c)
        print(" class name ", class_names[c])
        print(" AP: ", AP[i])
      #  print("Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    
   # print(f"mAP: {AP.mean()}")
    print("\n mAP: ", AP.mean())

#if __name__ == "__main__":
if __name__ == '__main__' and '__file__' in globals():
   multiprocessing.freeze_support()
   main()

#%%

if True:
  #  list_path=path
    counter=0
    print("creating files")
    basePath = "data/custom/images/"#file.readlines()[0]
    for animal in os.listdir(basePath):
        for img in os.listdir(basePath+"/"+animal):
            counter+=1
            if counter % 3:
                with open("train.txt", "a+") as file:
                    file.write(basePath+animal+"/"+img+"\n")
            else: 
                with open("valid.txt", "a+") as file:
                    file.write(basePath+animal+"/"+img+"\n")


    print("files created")
            
