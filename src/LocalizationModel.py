import torch
import torch.nn as nn
from HourGlass import hg1, hg2, hg8
from torch.hub import load_state_dict_from_url
import kornia as K
import math
import dsntnn
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import monai
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

def centroids_calc(coordinates):
    return torch.mean(coordinates, dim = 1)

def crop_coordinates(coords, img):
    
    centroids = centroids_calc(coords)
    
    x_min, x_max = coords[:, :, 0].min(dim = -1)[0], coords[:, :, 0].max(dim = -1)[0]
    y_min, y_max = coords[:, :, 1].min(dim = -1)[0], coords[:, :, 1].max(dim = -1)[0]
    

    w = (x_max - x_min) * 0.8#((0.9 - 0.7) * torch.rand(1, dtype = x_max.dtype, device = x_max.device) + 0.7)
    h = (y_max - y_min) * 1.3#((1.4 - 1.2) * torch.rand(1, dtype = x_max.dtype, device = x_max.device) + 1.2)
    
    x1, y1, x2, y2 = centroids[:, 0] - w, centroids[:, 1] - h, centroids[:, 0] + w, centroids[:, 1] + h
    
    y_in, x_in = img.shape[2:]
    #x1 and y1 must be at least zero so if they are < 0 they become 0

    x1[x1 < 0] = 0
    y1[y1 < 0] = 0
    x2[x2 > x_in] = x_in - 1
    y2[y2 > y_in] = y_in - 1

    return x1, y1, x2, y2

model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}

def _init_weights(model):
    for m in model.modules():
        if type(m) in {
            nn.Conv2d,
        }:
            nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_out', nonlinearity = 'relu')

            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)


class CoordRegressionNetwork(torch.nn.Module):
    #def __init__(self, arch, n_locations_global, n_locations_local, pretrained, kernel_size, n_ch, n_blocks):
    def __init__(self, arch, n_locations_global, n_locations_local, pretrained, n_ch, n_blocks):
        super().__init__()

        #self.kernel_size = kernel_size

        if arch == 'hg1':
            self.global_model = hg1(pretrained = False, num_classes = n_locations_global, num_blocks = n_blocks, n_ch = n_ch)
        elif arch == 'hg2':
            self.global_model = hg2(pretrained = False, num_classes = n_locations_global, num_blocks = n_blocks, n_ch = n_ch)
        else:
            self.global_model = hg8(pretrained = False, num_classes = n_locations_global, num_blocks = n_blocks, n_ch = n_ch)

        if pretrained:

            state_dict = load_state_dict_from_url(model_urls[arch], progress=True,
                                                    map_location=torch.device('cpu'))

            model_dict = self.global_model.state_dict()
            state_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
            }

            model_dict.update(state_dict)

            self.global_model.load_state_dict(model_dict)
        else:
            _init_weights(self.global_model)

        self.model_t1_c7 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        self.model_c7_c6 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        self.model_c6_c5 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        self.model_c5_c4 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        self.model_c4_c3 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        self.model_c3_c2 = hg1(pretrained = False, num_classes = n_locations_local, num_blocks = n_blocks, n_ch = n_ch)    
        
    def forward(self, x):
        avg_pool = nn.functional.avg_pool2d(x, self.kernel_size)
        hg_out = self.global_model(avg_pool)
        
        #appply spatial softmax to sum to 1 each channel
        heatmaps = K.geometry.subpix.spatial_softmax2d(hg_out[-1])
        #heatmaps = dsntnn.flat_softmax(hg_out[-1])

        #2D coords not normalized
        coords = K.geometry.subpix.spatial_soft_argmax2d(hg_out[-1], normalized_coordinates = True) 

        bbox_size = (128, 128)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ATTENTION TO THE TEST WITH THE TRUE COORDS
        coords_rec = dsntnn.normalized_to_pixel_coordinates(coords, bbox_size).float()
        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        with torch.no_grad():
            #coords of the corners
            c = [0, 1, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 20, 21, 18, 19, 24, 25]

            coords_reduced = coords_rec[:, c].reshape(x.shape[0], 6, 4, -1)

            bboxes = []
            for i in range(coords_reduced.shape[0]):
                x1, y1, x2, y2 = crop_coordinates(coords_reduced[i], x)
                bboxes.append(torch.stack((x1, y1, x2, y2), dim = 1))
        
        scale_factor = float(x.shape[-1] / bbox_size[0]) # must be 8 because the images are of shape 1024x1024 and boxes are scale 1024x1024
        box = torchvision.ops.roi_align(x, bboxes, output_size = bbox_size, spatial_scale = scale_factor)        
        #I want the boxes with shape batch x levels_boxes x 128 x 128
        box = box.reshape((x.shape[0], coords_reduced.shape[1], bbox_size[0], bbox_size[1]))

        #concatenate list of bboxes
        bboxes = torch.cat(bboxes)
        #reshape to have shape batch x levels_boxes x 4
        bboxes = bboxes.reshape((x.shape[0], coords_reduced.shape[1], -1))        
        
        #multiply by scale factor to have them in resolution 1024 x 1024. They are 128 x 128
        bboxes_rec = bboxes * scale_factor
        #coordinates of origin of bounding boxes
        origins = bboxes_rec[:, :, :2]
        #resolutions of bounding boxes orifginal
        dims = torch.stack(((bboxes_rec[:, :, 2] - bboxes_rec[:, :, 0]), (bboxes_rec[:, :, 3] - bboxes_rec[:, :, 1])), axis = 0).permute(1, 2, 0)
        #scale factor to use to convert to 128 x 128 
        dims = dims / bbox_size[0]

        #prediction with the local model
        hg_out_t1c7 = self.model_t1_c7(box[:, 0].unsqueeze(dim = 1))
        hg_out_c7c6 = self.model_c7_c6(box[:, 1].unsqueeze(dim = 1))
        hg_out_c6c5 = self.model_c6_c5(box[:, 2].unsqueeze(dim = 1))
        hg_out_c5c4 = self.model_c5_c4(box[:, 3].unsqueeze(dim = 1))
        hg_out_c4c3 = self.model_c4_c3(box[:, 4].unsqueeze(dim = 1))
        hg_out_c3c2 = self.model_c3_c2(box[:, 5].unsqueeze(dim = 1))
        
        # #2D coords normalized
        coords_t1c7 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_t1c7[-1], normalized_coordinates = True)
        coords_c7c6 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_c7c6[-1], normalized_coordinates = True)
        coords_c6c5 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_c6c5[-1], normalized_coordinates = True)
        coords_c5c4 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_c5c4[-1], normalized_coordinates = True)
        coords_c4c3 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_c4c3[-1], normalized_coordinates = True)
        coords_c3c2 = K.geometry.subpix.spatial_soft_argmax2d(hg_out_c3c2[-1], normalized_coordinates = True)

        # # coords in 128x128 resolution that is bboxes resolution
        coords_local = torch.cat((coords_t1c7, coords_c7c6, coords_c6c5, coords_c5c4, coords_c4c3, coords_c3c2), dim = 1)
        coords_local_bb = dsntnn.normalized_to_pixel_coordinates(coords_local, bbox_size).reshape(x.shape[0], bboxes.shape[1], 4, 2)

        # #points in org bbox dimensions
        coords_local = coords_local_bb * dims[:, :, None]           
        
        # #sum the coords of origin of bounding box
        coords_local = (coords_local + origins[:, :, None]).reshape(x.shape[0], -1, 2)      
        
        #back to range -1 1 for the loss
        coords_local = dsntnn.pixel_to_normalized_coordinates(coords_local, x.shape[2:])
        
        return coords, heatmaps, coords_local