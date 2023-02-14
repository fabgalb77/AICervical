# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:27:06 2021

@author: andre
"""

import torch
from torchvision import models
from tqdm import tqdm
from pathlib import Path
import copy
import numpy as np
import sys
import dsntnn
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

def block_to_freeze(model, number_of_block_to_freeze):
    
    model_freezed = model
    count = 0
    for child in model_freezed.children():
        if count < number_of_block_to_freeze:
            for param in child.parameters():
                param.requires_grad = False
        count += 1
    assert number_of_block_to_freeze < count, 'cannot freeze more than the number of blocks in the model'
    
    return model_freezed

def myResNet(number_of_block_to_freeze, pretrained = True):    
    resnet = models.resnet50(weights = ResNet50_Weights.DEFAULT)
    
    resnet_freezed = block_to_freeze(resnet, number_of_block_to_freeze)
    
    model = torch.nn.Sequential(*(list(resnet_freezed.children())[:-2]))
    
    num_ftrs = model[-1][-1].bn3.num_features

    return model, num_ftrs

class CoordRegressionNetworkResNet(torch.nn.Module):
    def __init__(self, model, n_locations, in_channel):
        super().__init__()
        
        self.fcn = model
        self.n_locations = n_locations
        self.in_channel = in_channel        
        
        self.hm_conv = torch.nn.Conv2d(self.in_channel, self.n_locations, kernel_size = 1, bias = False)
    
    def forward(self, x):
        
        fcn_out = self.fcn(x)
        
        unnormalized_heatmaps = self.hm_conv(fcn_out)
            
        hm = dsntnn.flat_softmax(unnormalized_heatmaps)
            
        coords = dsntnn.dsnt(hm)            
                
        return coords, hm