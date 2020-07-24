#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 07:55:50 2020

@author: dgatti
"""

# In[1]
# ### CONSTANTS

def _Params():

    # Image features
    # HEIGHT = 262
    # WIDTH = 400
    ### 3 GPUs ####
    # HEIGHT = 437
    # WIDTH = 667
    ###############
    HEIGHT = 300
    WIDTH = 300
    
    IMG_COLOR_MODE ='rgb'
    # IMG_COLOR_MODE = 'grayscale'
    
    if IMG_COLOR_MODE == 'rgb':
        CHANNELS = 3
    elif IMG_COLOR_MODE == 'grayscale':
        CHANNELS = 1
    
    IMG_CLASS = 'img'
    
    # Mask features
    # MSK_COLOR_MODE = 'rgb'
    # MSK_COLOR_MODE = 'rgba'
    MSK_COLOR_MODE = 'grayscale'
    NUM_CLASS = 4
    CLASSES = ['msk0','msk1','msk2','msk3']
    
    # Kernels 
    KS1 = (3,3) 
    KS2 = (5,5)
    KS3 = (7,7)
    # KS1 = (3,3) 
    # KS2 = (7,7)
    # KS3 = (11,11)    
    DL1 = (1,1)
    DL2 = (3,3)
    DL3 = (5,5)
    
    # Convolutional block Filters
    NF = 16
    # LSTM block Filters
    NFL = 1
    # NF = 24
    # Convolutional residual blocks
    NR1 = 6
    # LSTM residual blocks
    NR2 = 1
    
    # Dropout rates: 
    # DR1, conv blocks (recommended 0.05) 
    # DR2, LSTM blocks (recommended 0.1)
    DR1 = 0.05
    DR2 = 0.1
    
    # Residual block mode: add or concatenate different dilations
    DIL_MODE = "conc"
    # DIL_MODE = "add"
    
    # Weight mode: contour (recommended) or volume
    W_MODE = "contour"
    # W_MODE = "both"
    # W_MODE = "volume"
    
    # Loss_smoothing (recommended values: 1.0 for 'contour' mode, 1e-5 for 
    # 'volume' mode)
    if W_MODE == "contour":
        LS = 1
    elif W_MODE == "both":
        LS = 1        
    elif W_MODE == "volume":
        LS = 1e-5
    
    # Batch size for training and validation set
    TRAIN_SIZE = 10
    VAL_SIZE = 6
    TEST_SIZE = 6
    
    return HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, TEST_SIZE, DR1, DR2, CLASSES, IMG_CLASS

def _Paths():
    TRAIN_IMG_PATH = 'dataset/train_local/images'
    TRAIN_MSK_PATH = 'dataset/train_local/masks'
    VAL_IMG_PATH = 'dataset/val_local/images'
    VAL_MSK_PATH = 'dataset/val_local/masks'
    TEST_IMG_PATH = 'dataset/test_local/images'
    TEST_MSK_PATH = 'dataset/test_local/masks'    
    TRAIN_MSK_CLASS = ['msk']
    VAL_MSK_CLASS = ['msk']
    TEST_MSK_CLASS = ['msk']

    return TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS, TEST_IMG_PATH, TEST_MSK_PATH, TEST_MSK_CLASS

def _Seeds():
    TRAIN_SEED = 1
    VAL_SEED = 2
    TEST_SEED = 3
    return TRAIN_SEED, VAL_SEED, TEST_SEED
    
# In[2]
