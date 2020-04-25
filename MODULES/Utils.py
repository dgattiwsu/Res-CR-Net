#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:51:48 2020

@author: dgatti
"""

# In[1]
import numpy as np
import copy

# In[2]
def overlay_mask(image_layer,mask_layer,channel,fraction):
    image_layer = copy.deepcopy(image_layer)
    
    mask_layer = copy.deepcopy(mask_layer)
    mask_layer = mask_layer[:,:,channel]
    ind = mask_layer.astype(bool)
    
    if image_layer.shape[2] == 1:
        image_layer = np.squeeze(image_layer)        
        image_layer[ind] = image_layer[ind]*fraction 
        mask_layer = mask_layer*(1-np.max(image_layer[ind]))
        g_layer = image_layer + mask_layer
        rgb_layer = np.dstack((image_layer,g_layer,g_layer))
        
    elif image_layer.shape[2] == 3:
        r_layer = image_layer[:,:,0]
        g_layer = image_layer[:,:,1]
        b_layer = image_layer[:,:,2]

        g_layer[ind] = g_layer[ind]*fraction
        b_layer[ind] = b_layer[ind]*fraction
        mask_g_layer = mask_layer*(1-np.max(g_layer))
        mask_b_layer = mask_layer*(1-np.max(b_layer))

        g_layer = g_layer + mask_g_layer
        b_layer = b_layer + mask_b_layer

        rgb_layer = np.dstack((r_layer,g_layer,b_layer))
        
    return rgb_layer
    
def overlay_mask_2(image_layer,mask_layer,channel,fraction,mask_color):
    image_layer = copy.deepcopy(image_layer)    
    mask_layer = copy.deepcopy(mask_layer)
    ind = mask_layer.astype(bool)
    
    if image_layer.shape[2] == 1:
        image_layer = np.squeeze(image_layer)        
        image_layer[ind] = image_layer[ind]*fraction 
        mask_layer = mask_layer*(1-np.max(image_layer[ind]))
        g_layer = (image_layer + mask_layer)
        
        if mask_color == 'cyan':
            rgb_layer = np.dstack((image_layer,g_layer,g_layer))
        elif mask_color == 'yellow':
            rgb_layer = np.dstack((g_layer,g_layer,image_layer))
        elif mask_color ==  'violet':
            rgb_layer = np.dstack((g_layer,image_layer,g_layer))

        
    elif image_layer.shape[2] == 3:

        r_layer = np.squeeze(np.expand_dim(image_layer[:,:,0],axis=-1)) 
        g_layer = np.squeeze(np.expand_dim(image_layer[:,:,1],axis=-1)) 
        b_layer = np.squeeze(np.expand_dim(image_layer[:,:,2],axis=-1))            
                          
        if mask_color == 'cyan':
                   
            g_layer[ind] = g_layer[ind]*fraction 
            b_layer[ind] = b_layer[ind]*fraction
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            g_layer = g_layer + mask_layer            
            b_layer = b_layer + mask_layer            
           
        elif mask_color == 'yellow':
                   
            r_layer[ind] = r_layer[ind]*fraction 
            g_layer[ind] = g_layer[ind]*fraction
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            r_layer = r_layer + mask_layer            
            g_layer = g_layer + mask_layer
            
        elif mask_color ==  'violet':
                    
            g_layer[ind] = g_layer[ind]*fraction 
            b_layer[ind] = g_layer[ind]*fraction 
            mask_layer = mask_layer*(1-np.max(image_layer[ind]))
            r_layer = g_layer + mask_layer            
            b_layer = b_layer + mask_layer                                    
            
        rgb_layer = np.dstack((r_layer,g_layer,b_layer))
        
    return rgb_layer
    
def get_class_threshold(NUM_CLASS):
    step = 255/(NUM_CLASS-1)
    class_threshold = []
    for i in range(NUM_CLASS):
        jump = round(step*i)
        class_threshold.append(jump)
    return class_threshold