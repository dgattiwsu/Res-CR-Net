#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:33:07 2020

@author: dgatti
"""

# In[1]:

import os
import random
import numpy as np
import cv2

# In[2]:

from MODULES.Generators import train_generator_1, val_generator_1
from MODULES.Generators import train_generator_2, val_generator_2
from MODULES.Networks import ResNet_Atrous, Dense_ResNet_Atrous, Very_Dense_ResNet_Atrous
from MODULES.Networks import ResUNet, ResUNet_Big, ResUNet_CR, ResUNet_CR_Big
from MODULES.Losses import dice_coeff
from MODULES.Losses import tani_loss, tani_coeff, weighted_tani_coeff
from MODULES.Losses import weighted_tani_loss, other_metrics
from MODULES.Constants import _Params
from MODULES.Utils import get_class_threshold, get_model_memory_usage
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import datetime

# In[3]:

# ### MODEL SELECTION

HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params()
                
model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'

# In[4]:
    
# ### LOADING/COMPILATION

model_number = '2020-04-16_20_50'   
load_saved = True
load_best = True

if load_saved:
    # read in 
    json_file = open('models/' + model_selection + '_' + model_number + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
        
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = model_from_json(loaded_model_json)
        model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])        
        
    # load weights into new model
    if load_best:
        model.load_weights('models/best_' + model_selection + '_' + model_number + '_weights.h5')
    else:
        model.load_weights('models/last' + model_selection + '_' + model_number + '_weights.h5')      
    
# In[5]
    
# ### EVALUATION

if len(CLASSES) == 1:
    test_scores=model.evaluate(val_generator_1(),steps=1)
elif len(CLASSES) > 1:
    test_scores=model.evaluate(val_generator_2(),steps=1)    
print('Validation score = ',test_scores)

# In[6] 

# ### OTHER METRICS
if len(CLASSES) == 1:
    x_val,y_val = next(val_generator_1())
elif len(CLASSES) > 1:
    x_val,y_val = next(val_generator_2())

y_true = y_val
y_pred = model.predict(x_val)

tani = tani_coeff(y_true,y_pred).numpy()
dice, jaccard, precision, recall, f1_score = other_metrics(y_true,y_pred,0.5)
print('dice = ', dice)
print('tani = ', tani)
print('jaccard = ', jaccard)
print('precision = ', precision)
print('recall = ', recall)
print('F1 score = ', f1_score)

# In[7]

# ### PLOTS PATH
pwd = os.getcwd()
os.system('mkdir saved_images')
img_path = os.path.join(pwd,'saved_images/')
print(img_path)

class_threshold = get_class_threshold(NUM_CLASS)   
result = (y_pred > 0.5)*1 

# ### SELECT VALIDATION IMAGE

for ind in np.arange(x_val.shape[0]):
    print(ind)
    
    x = x_val[ind]
    y = y_val[ind]*class_threshold
    r = result[ind]*class_threshold     

    y = np.sum(y,axis=2)
    r = np.sum(r,axis=2)
    
    # Correction for r if using sigmoid instead of softmax activation 
    r[r>255] = 255      
    
    print(x.shape,y.shape,r.shape)
    
    # Save the ground_truth and predicted mask figure       
    cv2.imwrite(img_path + 'img_' + str(ind) + '_ground_truth_mask_' + model_number + '.png', y)
    cv2.imwrite(img_path + 'img_' + str(ind) + '_predicted_mask_' + model_number + '.png', r)
    
    # Save image + mask figure
    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    ax = fig.add_subplot(1, 3, 1)
    if x.shape[2] == 1:
        ax.imshow(np.squeeze(x), cmap="gray")
    elif x.shape[2] == 3: 
        ax.imshow(x)
        
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y, cmap="gray")    
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(r, cmap="gray")
    
    plt.savefig(img_path + 'img_' + str(ind) + '_hand_labelled_and_predicted_mask_' + model_number + '.png')
      
