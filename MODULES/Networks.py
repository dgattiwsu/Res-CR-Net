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
        
        # for cycle in range(resblock2):
        #     if cycle == 0:
        #         d2 = residual_convLSTM2D_block(d1,r_filters,num_class,rd=re_dropout)
        #     else:
        #         d2 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout) 
                
        if resblock2 > 0:
            for cycle in range(resblock2):
                if cycle == 0:
                    d2 = residual_convLSTM2D_block(d1,r_filters,num_class,rd=re_dropout)
                else:
                    d2 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout) 
        else:
            d2 = shrink_block(d1,num_class)
                                 
        outputs = Activation("softmax", name = 'softmax')(d2)
       
        # Optionally use sigmoid activation.
        # outputs = Activation("sigmoid", name = 'sigmoid')(d2)

        model = Model(inputs, outputs, name='Res-CR-Net')

        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])

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

        # for cycle in range(resblock2):
        #     if cycle == 0:
        #         d3 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout)
        #     else:
        #         d3 = residual_convLSTM2D_block(d3,r_filters,num_class,rd=re_dropout) 
                
        if resblock2 > 0:
            for cycle in range(resblock2):
                if cycle == 0:
                    d3 = residual_convLSTM2D_block(d2,r_filters,num_class,rd=re_dropout)
                else:
                    d3 = residual_convLSTM2D_block(d3,r_filters,num_class,rd=re_dropout) 
        else:
            d3 = shrink_block(d2,num_class)     
                                 
        outputs = Activation("softmax", name = 'softmax')(d3)
        
        # Optionally use sigmoid activation.
        # outputs = Activation("sigmoid", name = 'sigmoid')(d3)       

        # model = Model(inputs, outputs, name='Res-CRD-Net')

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
       
        # Optionally use sigmoid activation.
        # outputs = Activation("sigmoid", name = 'sigmoid')(d7)       

        model = Model(inputs, outputs, name='Res-CRD-Net')

        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])

    return model
   
# In[6]:
    
# ### Res-UR-NET
    
def ResUNet_CR(input_shape=(HEIGHT, WIDTH, CHANNELS),num_class=NUM_CLASS):
    f = [16, 32, 64, 128, 256]
    
#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        inputs = Input(shape=input_shape)
        
        ## Encoder    
        # e0 = select(inputs,0)
        e1 = stem_split(inputs, f[0])
        e2 = residual_block_split(e1, f[1], strides=2)
        e3 = residual_block_split(e2, f[2], strides=2)
        e4 = residual_block_split(e3, f[3], strides=2)
        e5 = residual_block_split(e4, f[4], strides=2)
        
        ## Bridge
        # b0 = conv_block(e5, f[4], strides=1)
        # b1 = conv_block(b0, f[4], strides=1)
        b1 = residual_block(e5, f[4])
        
        ## Decoder
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block_split(u1, f[4])
        
        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block_split(u2, f[3])
        
        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block_split(u3, f[2])
        
        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block_split(u4, f[1])
        
        # d5 = SeparableConv2D(num_class, (3, 3), padding="same", activation="relu", 
        #                      depthwise_initializer=he_normal(seed=5),
        #                      pointwise_initializer=he_normal(seed=5), 
        #                      bias_initializer='zeros')(d4)    
        
        d5 = Conv2D(num_class, (1, 1), padding="same", activation="relu", 
                    kernel_initializer=he_normal(seed=5), 
                    bias_initializer='zeros')(d4)     
    
        d6 = convLSTM2D_block(d5,1,num_class)
            
        outputs = Activation("softmax", name = 'softmax')(d6)
    
        model = Model(inputs, outputs, name='ResURNet')
        
        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])
    
    return model

# In[7]:

# ### RES-UR-NET BIG
    
def ResUNet_CR_Big(input_shape=(HEIGHT, WIDTH, CHANNELS),num_class=NUM_CLASS):
    f = [16, 32, 64, 128, 256, 512, 1024]
    
#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        inputs = Input(shape=input_shape)
        
        ## Encoder
        # e0 = select(inputs,0)
        e1 = stem_split(inputs, f[0])
        e2 = residual_block_split(e1, f[1], strides=2)
        e3 = residual_block_split(e2, f[2], strides=2)
        e4 = residual_block_split(e3, f[3], strides=2)
        e5 = residual_block_split(e4, f[4], strides=2)
        e6 = residual_block_split(e5, f[5], strides=2)
        e7 = residual_block_split(e6, f[6], strides=2)
        
        ## Bridge
        b0 = conv_block(e7, f[6], strides=1)
        b1 = conv_block(b0, f[6], strides=1)
        
        ## Decoder
        u1 = upsample_concat_block(b1, e6)
        d1 = residual_block_split(u1, f[6])
    
        u2 = upsample_concat_block(d1, e5)
        d2 = residual_block_split(u2, f[5])
        
        u3 = upsample_concat_block(d2, e4)
        d3 = residual_block_split(u3, f[4])
        
        u4 = upsample_concat_block(d3, e3)
        d4 = residual_block_split(u4, f[3])
    
        u5 = upsample_concat_block(d4, e2)
        d5 = residual_block_split(u5, f[2])
    
        u6 = upsample_concat_block(d5, e1)
        d6 = residual_block_split(u6, f[1])
    
        # d7 = SeparableConv2D(num_class, (3, 3), padding="same", activation="relu", 
        #                      depthwise_initializer=he_normal(seed=5),
        #                      pointwise_initializer=he_normal(seed=5), 
        #                      bias_initializer='zeros')(d6)
        
        d7 = Conv2D(num_class, (1, 1), padding="same", activation="relu", 
                    kernel_initializer=he_normal(seed=5), 
                    bias_initializer='zeros')(d6)
    
        d8 = convLSTM2D_block(d7,1,num_class)
    
        outputs = Activation("softmax", name = 'softmax')(d8)
        
        model = Model(inputs, outputs,name='ResURNet_Big')
        
        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])    
    
    return model

# In[8]:
    
# ### Res-U-NET 
    
def ResUNet(input_shape=(HEIGHT, WIDTH, CHANNELS),num_class=NUM_CLASS):
    f = [16, 32, 64, 128, 256]
    
#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        inputs = Input(shape=input_shape)
        
        ## Encoder    
        # e0 = select(inputs,0)
        e1 = stem_split(inputs, f[0])
        e2 = residual_block_split(e1, f[1], strides=2)
        e3 = residual_block_split(e2, f[2], strides=2)
        e4 = residual_block_split(e3, f[3], strides=2)
        e5 = residual_block_split(e4, f[4], strides=2)
        
        ## Bridge
        # b0 = conv_block(e5, f[4], strides=1)
        # b1 = conv_block(b0, f[4], strides=1)
        b1 = residual_block(e5, f[4])
        
        ## Decoder
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block_split(u1, f[4])
        
        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block_split(u2, f[3])
        
        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block_split(u3, f[2])
        
        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block_split(u4, f[1])
        
        # d5 = SeparableConv2D(num_class, (3, 3), padding="same", activation="relu", 
        #                      depthwise_initializer=he_normal(seed=5),
        #                      pointwise_initializer=he_normal(seed=5), 
        #                      bias_initializer='zeros')(d4)
        
        d5 = Conv2D(num_class, (1, 1), padding="same", activation="relu", 
                    kernel_initializer=he_normal(seed=5), 
                    bias_initializer='zeros')(d4)         
        
        outputs = Activation("softmax", name = 'softmax')(d5)
    
        model = Model(inputs, outputs, name='ResUNet')
        
        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])
    
    return model

# In[9]:

# ### RES-U-NET-BIG
    
def ResUNet_Big(input_shape=(HEIGHT, WIDTH, CHANNELS),num_class=NUM_CLASS):
    f = [16, 32, 64, 128, 256, 512, 1024]
    
#   tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        inputs = Input(shape=input_shape)
        
        ## Encoder
        # e0 = select(inputs,0)
        e1 = stem_split(inputs, f[0])
        e2 = residual_block_split(e1, f[1], strides=2)
        e3 = residual_block_split(e2, f[2], strides=2)
        e4 = residual_block_split(e3, f[3], strides=2)
        e5 = residual_block_split(e4, f[4], strides=2)
        e6 = residual_block_split(e5, f[5], strides=2)
        e7 = residual_block_split(e6, f[6], strides=2)
        
        ## Bridge
        b0 = conv_block(e7, f[6], strides=1)
        b1 = conv_block(b0, f[6], strides=1)
        
        ## Decoder
        u1 = upsample_concat_block(b1, e6)
        d1 = residual_block_split(u1, f[6])
    
        u2 = upsample_concat_block(d1, e5)
        d2 = residual_block_split(u2, f[5])
        
        u3 = upsample_concat_block(d2, e4)
        d3 = residual_block_split(u3, f[4])
        
        u4 = upsample_concat_block(d3, e3)
        d4 = residual_block_split(u4, f[3])
    
        u5 = upsample_concat_block(d4, e2)
        d5 = residual_block_split(u5, f[2])
    
        u6 = upsample_concat_block(d5, e1)
        d6 = residual_block_split(u6, f[1])
    
        # d7 = SeparableConv2D(num_class, (3, 3), padding="same", activation="relu", 
        #                      depthwise_initializer=he_normal(seed=5),
        #                      pointwise_initializer=he_normal(seed=5), 
        #                      bias_initializer='zeros')(d6)
        
        d7 = Conv2D(num_class, (1, 1), padding="same", activation="relu", 
                    kernel_initializer=he_normal(seed=5), 
                    bias_initializer='zeros')(d6)
    
        d8 = convLSTM2D_block(d7,1,num_class)
    
        outputs = Activation("softmax", name = 'softmax')(d8)
        
        model = Model(inputs, outputs,name='ResUNet_Big')
        
        # model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])    
    
    return model


