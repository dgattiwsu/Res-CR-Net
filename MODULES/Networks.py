#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:51:48 2020

@author: dgatti
"""

# In[1]
from MODULES.Blocks import * 
from MODULES.Constants import _Params
from MODULES.Losses import dice_coeff, tani_loss 
from MODULES.Losses import tani_coeff, weighted_tani_coeff
from MODULES.Losses import weighted_tani_loss
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, SeparableConv2D
from tensorflow.keras.layers import Conv2D, Concatenate, Add, LeakyReLU
from tensorflow.keras.layers import Dropout, SpatialDropout2D
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import Adam

# In[2]

# ### CONSTANTS
 
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()

# In[3]:    

# ### RESNET-ATROUS
    
def ResNet_Atrous(input_shape=(HEIGHT, WIDTH, CHANNELS),
                   num_class=NUM_CLASS,
                   ks1=KS1, ks2=KS2, ks3=KS3, 
                   dl1=DL1, dl2=DL2, dl3=DL3,
                   filters=NF,resblock1=NR1,
                   r_filters=NFL, resblock2=NR2,
                   dil_mode=DIL_MODE, 
                   sp_dropout=DR1,re_dropout=DR2):

#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        inputs = Input(shape=input_shape)

        for cycle in range(resblock1):
            if cycle == 0:
                d1 = stem_split_3k(inputs, filters, mode=dil_mode, 
                                   kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                   dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
            else:
                d1 = residual_block_split_3k(d1, filters, mode=dil_mode, 
                                             kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                             dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
            # d1 = Dropout(0.1)(d1)
            d1 = SpatialDropout2D(sp_dropout)(d1)
        
        for cycle in range(resblock2):
            if cycle == 0:
                d2 = residual_convLSTM2D_block(d1,r_filters,num_class,rd=re_dropout)
            else:
                d2 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout) 
      
        outputs = Activation("softmax", name = 'softmax')(d2)

        model = Model(inputs, outputs, name='Res-CR-Net')

        model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])

    return model

# In[4]:    

# ### DENSE-RESNET-ATROUS
    
def Dense_ResNet_Atrous(input_shape=(HEIGHT, WIDTH, CHANNELS),
                   num_class=NUM_CLASS,
                   ks1=KS1, ks2=KS2, ks3=KS3, 
                   dl1=DL1, dl2=DL2, dl3=DL3,
                   filters=NF,resblock1=NR1,
                   r_filters=NFL, resblock2=NR2,
                   dil_mode=DIL_MODE, 
                   sp_dropout=DR1,re_dropout=DR2):

#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        inputs = Input(shape=input_shape)

        for cycle in range(resblock1):
            if cycle == 0:
                d1 = stem_split_3k(inputs, filters, mode=dil_mode, 
                                   kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                   dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
            else:
                if cycle == 1:                        
                    d2 = residual_block_split_3k(d1, filters, mode=dil_mode, 
                                         kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                         dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
                    d2 = SpatialDropout2D(sp_dropout)(d2)
                    dsum = Add()([d1,d2])                    
                else:
                    d2 = residual_block_split_3k(d2, filters, mode=dil_mode, 
                                         kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                         dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)                        
                    d2 = SpatialDropout2D(sp_dropout)(d2)                
                    d2 += dsum
                    dsum = d2 + 0                                                          

        for cycle in range(resblock2):
            if cycle == 0:
                d3 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout)
            else:
                d3 = residual_convLSTM2D_block(d3,r_filters,num_class,rd=re_dropout)  
                                 
        outputs = Activation("softmax", name = 'softmax')(d3)

        model = Model(inputs, outputs, name='Res-CRD-Net')

        model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])

    return model
      
# In[5]:    

# ### VERY-DENSE-RESNET-ATROUS (6 CONV_RES_BLOCKS, 1 LSTM_RES_BLOCK)
    
def Very_Dense_ResNet_Atrous(input_shape=(HEIGHT, WIDTH, CHANNELS),
                   num_class=NUM_CLASS,
                   ks1=KS1, ks2=KS2, ks3=KS3, 
                   dl1=DL1, dl2=DL2, dl3=DL3,
                   filters=NF,resblock1=NR1,
                   r_filters=NFL,resblock2=NR2,
                   dil_mode=DIL_MODE, 
                   sp_dropout=DR1,re_dropout=DR2):

#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        inputs = Input(shape=input_shape)


        d1 = stem_split_3k(inputs, filters, mode=dil_mode, 
                                   kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                   dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)

        d2 = residual_block_split_3k(d1, filters, mode=dil_mode, 
                                     kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                     dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
        d2 = SpatialDropout2D(sp_dropout)(d2)                             
        d2 = Add()([d1, d2])


        d3 = residual_block_split_3k(d2, filters, mode=dil_mode, 
                                      kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                      dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
        d3 = SpatialDropout2D(sp_dropout)(d3)                                     
        d3 = Add()([d1, d2, d3])
        
        
        d4 = residual_block_split_3k(d3, filters, mode=dil_mode, 
                                      kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                      dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
        d4 = SpatialDropout2D(sp_dropout)(d4)                                      
        d4 = Add()([d1, d2, d3, d4])

                
        d5 = residual_block_split_3k(d4, filters, mode=dil_mode, 
                                      kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                      dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
        d5 = SpatialDropout2D(sp_dropout)(d5)                                      
        d5 = Add()([d1, d2, d3, d4, d5])

        
        d6 = residual_block_split_3k(d5, filters, mode=dil_mode, 
                                      kernel_size_1=ks1, kernel_size_2=ks2, kernel_size_3=ks3, 
                                      dilation_1=dl1, dilation_2=dl2, dilation_3=dl3)
        d6 = SpatialDropout2D(sp_dropout)(d6)                                      
        d6 = Add()([d1, d2, d3, d4, d5, d6])

                         
        d7 = residual_convLSTM2D_block(d6,r_filters,num_class,rd=re_dropout)
        
        # d1 = shrink_block(d1,num_class)
        # d2 = shrink_block(d2,num_class)
        # d3 = shrink_block(d3,num_class)
        # d4 = shrink_block(d4,num_class)
        # d5 = shrink_block(d5,num_class)
        # d6 = shrink_block(d6,num_class)           
        
        # d7 = Add()([d1, d2, d3, d4, d5, d6, d7])
      
        outputs = Activation("softmax", name = 'softmax')(d7)

        model = Model(inputs, outputs, name='Res-CRD-Net')

        model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])

    return model

