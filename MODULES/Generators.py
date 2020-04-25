#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 07:48:40 2020

@author: dgatti
"""

# In[1]:

import numpy as np
import tensorflow as tf
from MODULES.Constants import _Params, _Paths, _Seeds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[2]: 

# ### Generator functions for images and masks.

# CLASS_THRESHOLD      
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()

step = 255/(NUM_CLASS-1)
class_threshold = []
for i in range(NUM_CLASS):
    jump = round(step*i)
    class_threshold.append(jump)     

# This function converts a thresholded categorical gray scale mask into a standard
# categorical mask with consecutive indices [0,1,2,3,...]
    
def to_train_indices(x,threshold=class_threshold):
    x = np.floor(x)
    unique_values = np.unique(x)
    delta = np.ceil(255/(2*NUM_CLASS-2))
    x_mod = np.ones_like(x)*NUM_CLASS
    for i, val in enumerate(threshold):
        ind = (x > (val-delta)) & (x <= (val+delta))
        x_mod[ind] = i
    assert(np.max(x_mod)<NUM_CLASS)
    return x_mod 

def to_val_indices(x):
    x = np.floor(x)
    unique_values = np.unique(x)
    x_mod = np.zeros_like(x)
    for i, val in enumerate(unique_values):
        ind = x == val
        x_mod[ind] = i
    return x_mod 

def to_one_hot_train(x,b,h,w,num_class):
    unique_values = np.unique(x)
    x_out = np.zeros((b,h,w,num_class))
    for i, val in enumerate(unique_values):
        x_mod = np.zeros_like(x)
        ind = x == val
        x_mod[ind] = 1
        x_out[:,:,:,i] = x_mod[:,:,:,0]
        
    # The following pooling is used to smooth out the ragged edges of 
    # the mask after the preprocessing operations (rotation, shift, etc.).
    x_out = tf.round(tf.keras.backend.pool2d(x_out, pool_size=(3,3), \
                            strides=(1,1), padding='same', pool_mode='avg'))    
    return x_out
    
def to_one_hot_val(x,b,h,w,num_class):
    unique_values = np.unique(x)
    x_out = np.zeros((b,h,w,num_class))
    for i, val in enumerate(unique_values):
        x_mod = np.zeros_like(x)
        ind = x == val
        x_mod[ind] = 1
        x_out[:,:,:,i] = x_mod[:,:,:,0]
        
    return x_out
                   
               
# In[3]:

# ### TRAINING SET            
def train_generator_1():
    
    # ### CONSTANTS       
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
        
    TRAIN_SEED, VAL_SEED = _Seeds()
        
    train_data_gen_img_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,
                                 fill_mode='reflect')
                                 
    if MSK_COLOR_MODE == 'rgb':
        train_data_gen_msk_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
                                 
    elif MSK_COLOR_MODE == 'grayscale':
        train_data_gen_msk_args = dict(preprocessing_function=to_train_indices,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
    
    train_image_datagen = ImageDataGenerator(**train_data_gen_img_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_msk_args)
    
            
    train_image_generator = train_image_datagen.flow_from_directory(TRAIN_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                         
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                     
                                           seed=TRAIN_SEED)
    
    train_mask_generator = train_mask_datagen.flow_from_directory(TRAIN_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=CLASSES,
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                    
                                           seed=TRAIN_SEED)

    while True:
        if MSK_COLOR_MODE == 'rgb':
            yield(train_image_generator.next(), train_mask_generator.next())
        elif MSK_COLOR_MODE == 'grayscale':
            yield(train_image_generator.next(), to_one_hot_train(train_mask_generator.next(), \
                TRAIN_SIZE,HEIGHT,WIDTH,NUM_CLASS))

        
# In[4]:

# ### VALIDATION SET
def val_generator_1():

    # ### CONSTANTS     
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
        
    TRAIN_SEED, VAL_SEED = _Seeds()    
    
    val_data_gen_img_args = dict(rescale=1./255)
    
    if MSK_COLOR_MODE == 'rgb':
        val_data_gen_msk_args = dict(rescale=1./255)
        
    elif MSK_COLOR_MODE == 'grayscale':
        val_data_gen_msk_args = dict(preprocessing_function=to_val_indices)
    
    val_image_datagen = ImageDataGenerator(**val_data_gen_img_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_msk_args)
    
    
    val_image_generator = val_image_datagen.flow_from_directory(VAL_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                     
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED)
    
    val_mask_generator = val_mask_datagen.flow_from_directory(VAL_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=CLASSES,
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED)

    while True:
        if MSK_COLOR_MODE == 'rgb':
            yield(val_image_generator.next(), val_mask_generator.next())
        elif MSK_COLOR_MODE == 'grayscale':
            yield(val_image_generator.next(), to_one_hot_val(val_mask_generator.next(), \
                VAL_SIZE,HEIGHT,WIDTH,NUM_CLASS))
                
# In[5]

# ### TRAINING SET for multiple masks            
def train_generator_2():
    
    global mask
    
    # ### CONSTANTS       
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
    
    TRAIN_SEED, VAL_SEED = _Seeds()
        
    train_data_gen_img_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,
                                 fill_mode='reflect')
                                 

    train_data_gen_msk_args = dict(rescale=1./255,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=90,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=15,
                                 zoom_range=0.1,      
                                 fill_mode='reflect')
                                 
    
    train_image_datagen = ImageDataGenerator(**train_data_gen_img_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_msk_args)
    
            
    train_image_generator = train_image_datagen.flow_from_directory(TRAIN_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                         
                                           class_mode=None,
                                           batch_size=TRAIN_SIZE,
                                           shuffle=False,                     
                                           seed=TRAIN_SEED,
                                           # save_to_dir='dataset/train_local/images/save',
                                           )

    for i in range(NUM_CLASS):
        globals()['train_mask_gen_{}'.format(i)] = train_mask_datagen.flow_from_directory(TRAIN_MSK_PATH,
                                            target_size=(HEIGHT,WIDTH),
                                            classes=[CLASSES[i]],
                                            color_mode=MSK_COLOR_MODE,
                                            class_mode=None,
                                            batch_size=TRAIN_SIZE,
                                            shuffle=False,                    
                                            seed=TRAIN_SEED,
                                            # save_to_dir='dataset/train_local/masks/save',
                                            )
        
    while True:
        yield(train_image_generator.next(), \
              np.round(np.squeeze(np.stack([globals()['train_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3))))        
        # yield(train_image_generator.next(), \
        #       np.squeeze(np.stack([globals()['train_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3)))             
# In[6]:

# ### VALIDATION SET for multiple masks
def val_generator_2():

    # ### CONSTANTS     
    HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()
        
    TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS = _Paths()
    
    TRAIN_SEED, VAL_SEED = _Seeds()    
    
    val_data_gen_img_args = dict(rescale=1./255)
    val_data_gen_msk_args = dict(rescale=1./255)
            
    val_image_datagen = ImageDataGenerator(**val_data_gen_img_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_msk_args)
    
    
    val_image_generator = val_image_datagen.flow_from_directory(VAL_IMG_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[IMG_CLASS],
                                           color_mode=IMG_COLOR_MODE,                     
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED,
                                           # save_to_dir='dataset/val_local/images/save',
                                           )
    
    for i in range(NUM_CLASS):
        globals()['val_mask_gen_{}'.format(i)] = val_mask_datagen.flow_from_directory(VAL_MSK_PATH,
                                           target_size=(HEIGHT,WIDTH),
                                           classes=[CLASSES[i]],
                                           color_mode=MSK_COLOR_MODE,
                                           class_mode=None,
                                           batch_size=VAL_SIZE,
                                           shuffle=False,
                                           seed=VAL_SEED,
                                           # save_to_dir='dataset/val_local/masks/save',
                                           )
    
    while True:
        yield(val_image_generator.next(), \
              np.round(np.squeeze(np.stack([globals()['val_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3))))
        # yield(val_image_generator.next(), \
        #       np.squeeze(np.stack([globals()['val_mask_gen_{}'.format(i)].next() for i in range(NUM_CLASS)],axis=3)))              

        