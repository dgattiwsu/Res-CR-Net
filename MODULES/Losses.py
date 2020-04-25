#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:48 2020

@author: dgatti
"""

import tensorflow as tf
import numpy as np
from MODULES.Constants import _Params

# In[0]

# ### CONSTANTS
 
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()

# In[1]:
    
# ### LOSSES

# from tensorflow.keras.losses import binary_crossentropy
# import keras.backend as K
SHORT_AXIS = np.min([HEIGHT, WIDTH])

def weights(y_true, y_pred, w_mode=W_MODE, y_true_size=SHORT_AXIS):
    y_true = tf.cast(y_true, 'float32')
    
    if y_true_size <= 150:
        kernel_size = (11,11)
    elif y_true_size >150 and y_true_size <= 300:
        kernel_size = (21,21)
    elif y_true_size >300 and y_true_size <= 600:
        kernel_size = (31,31)
    elif y_true_size > 600:
        kernel_size = (41,41)    
    
    if w_mode == "contour": 
        # kernel size must be an odd number to get the same size of output
        averaged_mask = tf.keras.backend.pool2d(y_true, pool_size=kernel_size, strides=(1, 1), padding='same', pool_mode='avg')
        border = tf.cast(tf.greater(averaged_mask, 0.005), 'float32') * tf.cast(tf.less(averaged_mask, 0.995), 'float32')
        
        weight = tf.ones_like(averaged_mask)
        
        w0 = tf.reduce_sum(weight)
        weight += border * 2
        w1 = tf.reduce_sum(weight)
        weight *= (w0 / w1)
        weight = weight * weight
        
    elif w_mode == "both":
        averaged_mask = tf.keras.backend.pool2d(y_true, pool_size=kernel_size, strides=(1, 1), padding='same', pool_mode='avg')
        border = tf.cast(tf.greater(averaged_mask, 0.005), 'float32') * tf.cast(tf.less(averaged_mask, 0.995), 'float32')
        
        weight = tf.reduce_mean(tf.reduce_sum(y_true,axis=[1,2]),axis=0)  
        weight = tf.math.reciprocal(weight**2)
        scale = tf.tensordot(weight,tf.ones_like(weight),[0,0]) / tf.tensordot(weight,weight,[0,0])
        weight *= scale*tf.ones_like(y_true)
        
        w0 = tf.reduce_sum(weight)
        weight += border * 2
        w1 = tf.reduce_sum(weight)
        weight *= (w0 / w1)
        weight = weight * weight    
        
    elif w_mode == "volume": 
        weight = tf.reduce_mean(tf.reduce_sum(y_true,axis=[1,2]),axis=0)  
        weight = tf.math.reciprocal(weight**2)
        scale = tf.tensordot(weight,tf.ones_like(weight),[0,0]) / tf.tensordot(weight,weight,[0,0])        
        weight *= scale*tf.ones_like(y_true)
        
    return weight 

def dice_coeff(y_true, y_pred, smooth=LS):
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (2*intersection + smooth) / ((tf.reduce_sum(y_true) + tf.reduce_sum(tf.ones_like(y_true)*y_pred)) + smooth)
    return score
        
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def cce_dice_loss(y_true, y_pred):
    loss = tf.keras.backend.categorical_crossentropy(y_true,y_pred,from_logits=False) + dice_loss(y_true, y_pred)
    return loss

# Tanimoto coefficient without complement
def tani_coeff_nc(y_true, y_pred, smooth=LS):
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + smooth) / (tf.reduce_sum((y_true**2 + y_pred**2 - y_true * y_pred)) + smooth)
    return score
    
# Tanimoto coefficient with complement
def tani_coeff(y_true, y_pred, smooth=LS):
    y_true_c = 1 - y_true
    y_pred_c = 1 - y_pred    
    
    intersection_d = tf.reduce_sum(y_true * y_pred)
    score_d = (intersection_d + smooth) / (tf.reduce_sum((y_true**2 + y_pred**2 - y_true * y_pred)) + smooth)

    intersection_c = tf.reduce_sum(y_true_c * y_pred_c)
    score_c = (intersection_c + smooth) / (tf.reduce_sum((y_true_c**2 + y_pred_c**2 - y_true_c * y_pred_c)) + smooth)

    score = (score_d + score_c)/2   
    return score
    
def tani_loss(y_true, y_pred):
    loss = 1 - tani_coeff(y_true, y_pred)
    return loss
    
def cce_tani_loss(y_true, y_pred):
    loss = tf.keras.backend.categorical_crossentropy(y_true,y_pred,from_logits=False) + tani_loss(y_true, y_pred)
    return loss

def weighted_dice_coeff(y_true, y_pred, smooth=LS):
    w = weights(y_true, y_pred)
    intersection = (y_true * y_pred)    
    score = (2. * tf.reduce_sum(w * intersection) + smooth) / (tf.reduce_sum(w * y_true) + tf.reduce_sum(w * y_pred) + smooth)
    return score

def weighted_tani_coeff(y_true, y_pred, smooth=LS):
    y_true_c = 1 - y_true
    y_pred_c = 1 - y_pred
    w = weights(y_true, y_pred)
    
    intersection_d = tf.reduce_sum(w * y_true * y_pred)
    score_d = (intersection_d + smooth) / (tf.reduce_sum(w * (y_true**2 + y_pred**2 - y_true * y_pred)) + smooth)

    intersection_c = tf.reduce_sum(w * y_true_c * y_pred_c)
    score_c = (intersection_c + smooth) / (tf.reduce_sum(w * (y_true_c**2 + y_pred_c**2 - y_true_c * y_pred_c)) + smooth)

    score = (score_d + score_c)/2   
    return score

def weighted_dice_loss(y_true, y_pred):
    loss = 1 - weighted_dice_coeff(y_true, y_pred)
    return loss

def weighted_tani_loss(y_true, y_pred):
    loss = 1 - weighted_tani_coeff(y_true, y_pred)
    return loss

def weighted_bce_loss(y_true, y_pred):
    w = weights(y_true, y_pred)
    # avoiding overflow
    epsilon = 1e-7
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = tf.keras.backend.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (w - 1.) * y_true) *  \
    (tf.keras.backend.log(1. + tf.exp(-tf.abs(logit_y_pred))) + tf.maximum(-logit_y_pred, 0.))
    return tf.reduce_sum(loss) / tf.reduce_sum(w)

def weighted_bce_dice_loss(y_true, y_pred):
    loss = weighted_bce_loss(y_true, y_pred) + (1 - weighted_dice_coeff(y_true, y_pred))
    return loss

def weighted_bce_tani_loss(y_true, y_pred):
    loss = weighted_bce_loss(y_true, y_pred) + (1 - weighted_tani_coeff(y_true, y_pred))
    return loss

def cce_loss(y_true, y_pred):       
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    epsilon = 1e-7
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)    
    loss = y_true * tf.keras.backend.log(y_pred)
    loss = -tf.reduce_mean(loss)
    return loss
    
def weighted_cce_loss(y_true, y_pred):
    weight = weights(y_true, y_pred)
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    epsilon = 1e-7
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)    
    loss = y_true * tf.keras.backend.log(y_pred) * weight
    loss = -tf.reduce_mean(loss)
    return loss

def weighted_cce_tani_loss(y_true, y_pred):
    # WEIGHTS
    w = weights(y_true, y_pred)
    
    # WEIGHTED CROSS_ENTROPY
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    epsilon = 1e-7
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)    

    weighted_cce_loss = y_true * tf.keras.backend.log(y_pred) * w
    weighted_cce_loss = -tf.reduce_mean(weighted_cce_loss)        

    # WEIGHTED TANI COEFFICIENT
    smooth = 1.0e-5
    y_true_c = 1 - y_true
    y_pred_c = 1 - y_pred
        
    intersection_d = tf.reduce_sum(w * y_true * y_pred)
    score_d = (intersection_d + smooth) / (tf.reduce_sum(w * (y_true**2 + y_pred**2 - y_true * y_pred)) + smooth)

    intersection_c = tf.reduce_sum(w * y_true_c * y_pred_c)
    score_c = (intersection_c + smooth) / (tf.reduce_sum(w * (y_true_c**2 + y_pred_c**2 - y_true_c * y_pred_c)) + smooth)

    weighted_tani_coeff = (score_d + score_c)/2   
            
    # WEIGHTED TANI LOSS
    weighted_tani_loss = 1 - weighted_tani_coeff
    
    loss = weighted_cce_loss + weighted_tani_loss
    return loss

# Use the following to evaluate predictions after training:
# def bce_dice_loss_2(y_true,y_pred):
#     loss = tf.reduce_mean(binary_crossentropy(y_true,y_pred,from_logits=False)) + dice_loss(y_true, y_pred)
#     return loss

from tensorflow.keras.losses import categorical_crossentropy
def bce_dice_loss_2(y_true,y_pred):
    loss = tf.reduce_mean(categorical_crossentropy(y_true,y_pred,from_logits=False)) + dice_loss(y_true, y_pred)
    return loss

def dice_coeff_corr(y_true, y_pred, smooth=LS):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    dice_union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred[y_true.astype(bool)])
    dice_coeff =  (2. * intersection + smooth) / (dice_union + smooth)
    dice_sum = tf.reduce_sum(y_pred[0,:,:,:])/(HEIGHT*WIDTH)
    dice_max =  np.max((y_pred[0,:,:,2]))
    return dice_union, dice_coeff, dice_sum, dice_max
    
# ### OTHER METRICS

def other_metrics(y_true,y_pred,threshold):
    y_pred = (y_pred > threshold)*1
    TP = tf.reduce_sum(y_true * y_pred).numpy()
    FP = (tf.reduce_sum(y_pred) - TP).numpy()
    FN = (tf.reduce_sum(y_true) - TP).numpy()
    dice = 2*TP/(2*TP + FP + FN)
    jaccard = TP/(TP + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = (2 * precision * recall)/(precision + recall)
    return dice, jaccard, precision, recall, f1_score   
