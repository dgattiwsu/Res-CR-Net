#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:26:26 2020

@author: dgatti
"""

# In[1]:

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import he_normal, glorot_uniform, orthogonal
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, Activation 
from tensorflow.keras.layers import SeparableConv2D, Add, Concatenate, Bidirectional
from tensorflow.keras.layers import ConvLSTM2D, LeakyReLU

# In[1]:

# ### NETWORK BLOCKS

def stem_split_3k(x, filters, mode="conc",
                 kernel_size_1=(3,3), 
                 kernel_size_2=(5,5), 
                 kernel_size_3=(7,7),
                 dilation_1=(1,1),
                 dilation_2=(1,1), 
                 dilation_3=(1,1),
                 padding="same",strides=1):

    if mode == "conc":
        res_filters = filters
        skip_filters = np.uint64(filters*3)
    elif mode == "add":
        res_filters = filters
        skip_filters = filters
    
    x1 = SeparableConv2D(res_filters, kernel_size=kernel_size_1,
                         dilation_rate=dilation_1,
                         padding=padding, strides=1, 
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x)
    x1 = BatchNormalization()(x1)
    # x1 = Activation("relu")(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    
    res1 = SeparableConv2D(res_filters, kernel_size=kernel_size_1,
                           dilation_rate=dilation_1,
                           padding=padding, strides=1, 
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x1)
        
    x2 = SeparableConv2D(res_filters, kernel_size=kernel_size_2,
                         dilation_rate=dilation_2,
                         padding=padding, strides=1, 
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x)
    x2 = BatchNormalization()(x2)
    # x2 = Activation("relu")(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    
    res2 = SeparableConv2D(res_filters, kernel_size=kernel_size_2,
                           dilation_rate=dilation_2,
                           padding=padding, strides=1, 
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x2)      

    x3 = SeparableConv2D(res_filters, kernel_size=kernel_size_3,
                         dilation_rate=dilation_3,
                         padding=padding, strides=1, 
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x)
    x3 = BatchNormalization()(x3)
    # x3 = Activation("relu")(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    
    res3 = SeparableConv2D(res_filters, kernel_size=kernel_size_3,
                           dilation_rate=dilation_3,
                           padding=padding, strides=1, 
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x3)      
    
    shortcut = Conv2D(skip_filters, kernel_size=(1,1), padding=padding, 
                      strides=1, 
                      kernel_initializer=he_normal(seed=5),
                      bias_initializer='zeros')(x)    
    
    shortcut = BatchNormalization()(shortcut)
    
    if mode == "conc":
        res = Concatenate()([res1, res2, res3])
        output = Add()([shortcut, res])
    elif mode == "add":
        output = Add()([shortcut, res1, res2, res3])
    
    return output

def residual_block_split_3k(x, filters, mode="conc", 
                           kernel_size_1=(3,3),
                           kernel_size_2=(5,5),
                           kernel_size_3=(7,7),
                           dilation_1=(1,1),
                           dilation_2=(1,1), 
                           dilation_3=(1,1),
                           padding="same",strides=1):    

    if mode == "conc":
        res_filters = filters
        skip_filters = np.uint64(filters*3)
    elif mode == "add":
        res_filters = filters
        skip_filters = filters
    
    x0 = BatchNormalization()(x)    
    # x0 = Activation("relu")(x0)
    x0 = LeakyReLU(alpha=0.1)(x0)    

    x1 = SeparableConv2D(res_filters, kernel_size=kernel_size_1,
                         dilation_rate=dilation_1,
                         padding=padding, 
                         strides=1, 
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x0)
    x1 = BatchNormalization()(x1)
    # x1 = Activation("relu")(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    
    res1 = SeparableConv2D(res_filters, kernel_size=kernel_size_1,
                           dilation_rate=dilation_1,
                           padding=padding, 
                           strides=1,
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x1)
        
    x2 = SeparableConv2D(res_filters, kernel_size=kernel_size_2,
                         dilation_rate=dilation_2,
                         padding=padding, 
                         strides=1,
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x0)
    x2 = BatchNormalization()(x2)
    # x2 = Activation("relu")(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    
    res2 = SeparableConv2D(res_filters, kernel_size=kernel_size_2,
                           dilation_rate=dilation_2, 
                           padding=padding, 
                           strides=1,
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x2)      

    x3 = SeparableConv2D(res_filters, kernel_size=kernel_size_3,
                         dilation_rate=dilation_3,
                         padding=padding, 
                         strides=1,
                         depthwise_initializer=he_normal(seed=5),
                         pointwise_initializer=he_normal(seed=5), 
                         bias_initializer='zeros')(x0)
    x3 = BatchNormalization()(x3)
    # x3 = Activation("relu")(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    
    res3 = SeparableConv2D(res_filters, kernel_size=kernel_size_3,
                           dilation_rate=dilation_3,
                           padding=padding, 
                           strides=1,
                           depthwise_initializer=he_normal(seed=5),
                           pointwise_initializer=he_normal(seed=5), 
                           bias_initializer='zeros')(x3)      
    
    shortcut = Conv2D(skip_filters, kernel_size=(1,1), padding=padding, 
                      strides=1, 
                      kernel_initializer=he_normal(seed=5),
                      bias_initializer='zeros')(x)        
    
    shortcut = BatchNormalization()(shortcut)

    if mode == "conc":
        res = Concatenate()([res1, res2, res3])
        output = Add()([shortcut, res])
    elif mode == "add":
        output = Add()([shortcut, res1, res2, res3])    
    
    return output


def residual_convLSTM2D_block(x,filters,num_class,rd=0.1):       

    x = Conv2D(num_class, kernel_size=(1,1), padding="same", strides=1,
                kernel_initializer=he_normal(seed=5), 
                bias_initializer='zeros')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    o2 = Lambda(lambda x: x[:,:,:,:,tf.newaxis])(x) 
    o3 = Bidirectional(ConvLSTM2D(filters=filters,kernel_size=(3,num_class),padding='same',
                                  kernel_initializer=he_normal(seed=5), 
                                  recurrent_initializer=orthogonal(gain=1.0, seed=5),
                                  activation='tanh',
                                  return_sequences=True,recurrent_dropout=rd))(o2)
     
    o2t = tf.transpose(o2,perm=[0, 2, 1, 3, 4])
    
    o3t = Bidirectional(ConvLSTM2D(filters=filters,kernel_size=(3,num_class),padding='same',
                                   kernel_initializer=he_normal(seed=5), 
                                   recurrent_initializer=orthogonal(gain=1.0, seed=5),
                                   activation='tanh',
                                   return_sequences=True,recurrent_dropout=rd))(o2t)
      
    o3t = tf.transpose(o3t,perm=[0, 2, 1, 3, 4]) 

    o4 = Add()([o3, o3t])    
    res = tf.reduce_sum(o4,axis=-1)
    
    shortcut = Conv2D(num_class, kernel_size=(1,1), padding='same', 
                      strides=1, 
                      kernel_initializer=he_normal(seed=5),
                      bias_initializer='zeros')(x)        
    
    shortcut = LeakyReLU(alpha=0.1)(shortcut)   
    
    output = Add()([x, res])
    
    return output
    

def shrink_block(x,num_class):
    
    x = Conv2D(num_class, kernel_size=(1,1), padding="same", strides=1,
                kernel_initializer=he_normal(seed=5), 
                bias_initializer='zeros')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    return x       
    

# In[5]: