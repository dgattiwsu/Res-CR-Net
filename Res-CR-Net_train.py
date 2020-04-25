#/Users/dgatti/venv_jupyter/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:51:48 2020

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
from MODULES.Networks import ResNet_Atrous, Dense_ResNet_Atrous
from MODULES.Losses import dice_coeff
from MODULES.Losses import tani_loss, tani_coeff, weighted_tani_coeff
from MODULES.Losses import weighted_tani_loss, other_metrics
from MODULES.Constants import _Params
from MODULES.Utils import get_class_threshold
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json 
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import datetime

# In[3]:

# ### DEVICES

physical_devices_GPU = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices_GPU))

physical_devices_CPU = tf.config.list_physical_devices('CPU') 
print("Num CPUs:", len(physical_devices_CPU)) 

local_device_protos = device_lib.list_local_devices()
print(local_device_protos)   

# In[4]:

# ### MODEL SELECTION
HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
    KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
    TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS = _Params() 

model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'

model_number = str(datetime.datetime.now())[0:10] + '_' + \
               str(datetime.datetime.now())[11:13] + '_' + \
               str(datetime.datetime.now())[14:16]

model = ResNet_Atrous()
print('ResNet_Atrous model selected')

# model = Dense_ResNet_Atrous()
# print('ResNet_Atrous model selected')

# In[5]

# ### SUMMARY

model.summary()

# In[6]
# Save architecture without weights as h5
model.save('models/' + model_selection + '_' + model_number + '.h5')

# ### Save architecture as json
model_json = model.to_json()
with open('models/' + model_selection + '_' + model_number + '.json', "w") as json_file:
    json_file.write(model_json) 

# In[7]

# ### MODEL GRAPH
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True,\
           show_layer_names=False,\
           to_file='saved_images/' + model_selection + '_' + model_number + '_architecture.png') 

# In[8]:
    
# # ### LOADING FROM CHECKPOINT

# model_number = '2020-03-27_11_18'   
# load_saved = True
# load_best = True

# if load_saved:
#     # read in 
#     json_file = open('models/' + model_selection + '_' + model_number + '.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into new model
#     if load_best:
#         model.load_weights('models/best_' + model_selection + '_' + model_number + '_weights.h5')
#     else:
#         model.load_weights('models/' + model_selection + '_' + model_number + '_weights.h5')
        
# # Get new model number if appropriate and save the model  
# model_number = str(datetime.datetime.now())[0:10] + '_' + \
#                 str(datetime.datetime.now())[11:13] + '_' + \
#                 str(datetime.datetime.now())[14:16] 

# # Save architecture as json
# model_json = model.to_json()
# with open('models/' + model_selection + '_' + model_number + '.json', "w") as json_file:
#     json_file.write(model_json)
        
# ### COMPILATION/EVALUATION
# model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])    
# test_scores=model.evaluate(val_generator(),steps=1)  
# print('Validation score = ',test_scores) 
    
# In[9]:

# ### CALLBACKS
               
# Quiery the current directory
pwd = os.getcwd()
os.system('mkdir log_dir')
os.system('rm -rf log_dir/*')

log_dir = pwd + '/log_dir'
print(log_dir)

callbacks_list = [
# callback for logging the history
    tf.keras.callbacks.CSVLogger('models/' + model_selection + \
        '_' + model_number + '_history.csv', append=True),
            
# callback for tensorboard: launch from a different venv (not ipython) console as 
# 'tensorboard --logdir log_dir'
    tf.keras.callbacks.TensorBoard( 
        log_dir=log_dir,
        histogram_freq=0, 
        write_graph=True, 
        write_images=True,
        profile_batch=0),
    
# callback for early stopping     
    tf.keras.callbacks.EarlyStopping( 
        monitor='val_loss', 
        patience=60, 
#         verbose=1, 
        min_delta = 0.001,
        mode='min'),

# callback to save best model (here we save only the weights because the 
# dice coef loss is not stored in the model)    
    tf.keras.callbacks.ModelCheckpoint( 
        filepath='models/best_' + model_selection + '_' + model_number + '_weights.h5',
        monitor='val_loss', save_best_only=True,
        mode = 'min', 
#         verbose = 1,
        save_weights_only = True)]
   
# In[10]:

# ### TRAINING   

epoch_num = 90
train_steps = 30 # Number of batches called in each epoch
val_steps = 1
  
if len(CLASSES) == 1:  
    history = model.fit(train_generator_1(), 
                        validation_data=val_generator_1(), 
                        steps_per_epoch=train_steps, 
                        validation_steps=val_steps, 
                        epochs=epoch_num,
                        callbacks = callbacks_list)
elif len(CLASSES) > 1:
    history = model.fit(train_generator_2(), 
                        validation_data=val_generator_2(), 
                        steps_per_epoch=train_steps, 
                        validation_steps=val_steps, 
                        epochs=epoch_num,
                        callbacks = callbacks_list)    
 
# In[11]:

# ### SAVE WEIGHTS

model.save_weights('models/' + model_selection + '_' + model_number + '_weights' + '.h5')

# In[12]:

# ### HISTORY

history.history.keys()
skip = 1
loss_history = history.history['loss']
dice_coeff_history = history.history['tani_coeff']
val_loss_history = history.history['val_loss']
val_dice_coeff_history = history.history['val_tani_coeff']
epochs = range(0, len(loss_history) , skip) 
plt.plot(epochs, loss_history[0::skip], '-', label='Training loss') 
plt.plot(epochs, dice_coeff_history[0::skip], '-', label='Training Tani coef') 
plt.plot(epochs, val_loss_history[0::skip], '--', label='Validation loss')
plt.plot(epochs, val_dice_coeff_history[0::skip], '-r', label='Validation Tani coef')
plt.title('Training and validation loss ') 
plt.xlabel('Epochs') 
plt.ylabel('Loss/Tani coefficient') 
plt.legend()
plt.grid()
plt.gcf()
plt.savefig('models/' + model_selection + '_' + model_number + '_history.png')

# In[13]:

# ### EVALUATION 

if len(CLASSES) == 1:
    test_scores=model.evaluate(val_generator_1(),steps=1)
elif len(CLASSES) > 1:
    test_scores=model.evaluate(val_generator_2(),steps=1)
    
print('Validation score = ',test_scores)

# ### LOADING/COMPILATION

# model_number = '2020-04-03_16_59'   
load_saved = True
load_best = True

if load_saved:
    # read in 
    json_file = open('models/' + model_selection + '_' + model_number + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    if load_best:
        model.load_weights('models/best_' + model_selection + '_' + model_number + '_weights.h5')
    else:
        model.load_weights('models/' + model_selection + '_' + model_number + '_weights.h5')
    
# In[14]
    
# ### EVALUATION
model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])
if len(CLASSES) == 1:
    test_scores=model.evaluate(val_generator_1(),steps=1)
elif len(CLASSES) > 1:
    test_scores=model.evaluate(val_generator_2(),steps=1)
    
print('Validation score = ',test_scores)


# In[15]

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

# In[16]

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
      
