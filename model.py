import pandas as pd
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from helper import batch_generator


# reading 1st scenario backward data
root_folter = './data/'
data_Folder = root_folter+'data_1st_scenario_backward/driving_log.csv'
df_1st_bwd = pd.read_csv(data_Folder, names=['center image path', 'left image path', 'right image path', 
                                                'steering angle', 'Throttle value','Break value',
                                               'speed'])
# reading 1st scenario forward data
data_Folder = root_folter+'data_1st_scenario_forward/driving_log.csv'
df_1st_fwd = pd.read_csv(data_Folder, names=['center image path', 'left image path', 'right image path', 
                                                'steering angle', 'Throttle value','Break value',
                                               'speed'])
# reading 2nd scenario forward data
data_Folder = root_folter+'dataset_2nd_scenario_forward/driving_log.csv'
df_2nd_fwd = pd.read_csv(data_Folder, names=['center image path', 'left image path', 'right image path', 
                                                'steering angle', 'Throttle value','Break value',
                                               'speed'])

## Get rid of some data since most of the steering angles are around zero
high_limit = 500
# 1st scenario backward
number_to_drop = len(df_1st_bwd[df_1st_bwd['steering angle']==0.0])-high_limit
df_1st_bwd.drop(df_1st_bwd[df_1st_bwd['steering angle']==0.0].sample(number_to_drop).index, inplace=True)
# 1st scenario forward
number_to_drop = len(df_1st_fwd[df_1st_fwd['steering angle']==0.0])-high_limit
df_1st_fwd.drop(df_1st_fwd[df_1st_fwd['steering angle']==0.0].sample(number_to_drop).index, inplace=True)
# 2nd scenario forward
number_to_drop = len(df_2nd_fwd[df_2nd_fwd['steering angle']==0.0])-high_limit
df_2nd_fwd.drop(df_2nd_fwd[df_2nd_fwd['steering angle']==0.0].sample(number_to_drop).index, inplace=True)

# Joining the dataframes into one
df = pd.concat([df_1st_bwd, df_1st_fwd, df_2nd_fwd], ignore_index=True)

## Let's 1st split data into training and validation
train_df, validation_df = train_test_split(df, test_size=0.2)


## Creating generators¶
batch_size = 10
val_generator = batch_generator(validation_df, batch_size)
train_generator = batch_generator(train_df, batch_size)

## Build model

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# cropping the model to get rid of unnecessary background
model.add(Cropping2D(cropping=((60,20), (0,0))))
# Start architecture here
model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

## Callbacks, Loss, optimizer setup
model_path="model.h5"

checkpoint = ModelCheckpoint(model_path, 
                              monitor= 'val_loss', 
                              verbose=1, 
                              save_best_only=True, 
                              mode= 'min', 
                              save_weights_only = False,
                              period=3)

early_stop = EarlyStopping(monitor='val_loss', 
                       mode= 'min', 
                       patience=2)


callbacks_list = [checkpoint, early_stop]

model.compile(loss='mse', optimizer='adam')

# start training
history=model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_df)/batch_size), 
            validation_data=val_generator, 
            validation_steps=np.ceil(len(validation_df)/batch_size), 
            epochs=5, verbose=1, callbacks=callbacks_list)
