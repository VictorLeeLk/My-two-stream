#-*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader

if __name__ == '__main__':

    rgb_preds='record/spatial/spatial_video_preds_resnet50.pickle'
    opf_preds = 'record/motion/motion_video_preds_resnet50.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path='/media/victorleelk/Elements/Keras_Project_Dataset/datasets/jpegs_256/', 
                                    ucf_list='/home/victorleelk/two-stream-from-github/two-stream-action-recognition-master/UCF_list/',
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),101))  # len(rgb.keys()) =  3783

    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]  # r: every video's 101 class value
        o = opf[name]  # o: every video's 101 class value

        label = int(test_video[name])-1
                    
        video_level_preds[ii,:] = (r+o)
        #video_level_preds[ii,:] = max(r,o)
        #for i in range(101):
        #   video_level_preds[ii,i]=2*r[i]+o[i]
        """
            if r[i]>o[i]:
                video_level_preds[ii,i]=r[i]
            else:
                video_level_preds[ii,i]=o[i]
        """

        video_level_labels[ii] = label
        ii+=1         
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                                
    print 'top1=',top1,'top5=',top5
