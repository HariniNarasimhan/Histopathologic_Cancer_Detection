import pandas as pd
import numpy as np
import keras
import os
import shutil
import skimage.io as skio
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
import matplotlib.pyplot as plt
import tensorflow as tf

file = pd.read_csv("train_labels.csv")

file =file[file['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2'] 
file =file[file['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe'] 

train,valid = train_test_split(file,test_size =0.33, random_state=42)

print(train.shape)
print(valid.shape)

batch_size = XXX
epochs = XXX
print("----------------------------------------------------")
print(train.head())
print("----------------------------------------------------")
def create_folder(folderName):
   if not os.path.exists(folderName):
       try:
           os.makedirs(folderName)
       except OSError as exc:
           if exc.errno != errno.EEXIST:
               raise
base_dir = 'data'
create_folder(base_dir)
train_dir = os.path.join(base_dir,'train_dataset')
create_folder(train_dir)
valid_dir = os.path.join(base_dir,'valid_dataset')
create_folder(valid_dir)

train_tum = os.path.join(train_dir,'0')
create_folder(train_tum)
train_notum = os.path.join(train_dir,'1')
create_folder(train_notum)

valid_tum = os.path.join(valid_dir,'0')
create_folder(valid_tum)

valid_notum = os.path.join(valid_dir,'1')
create_folder(valid_notum)

for images in range(0,len(train)):
   file_name = train.iloc[images].values[0] + '.tif'
   y_train = train.iloc[images].values[1]
   if(y_train == 0):
       train_path = train_tum
   else:
       train_path = train_notum
   src = os.path.join('dataset/train/' , file_name)
   dest = os.path.join(train_path , file_name)
   shutil.copyfile(src,dest)

for images in range(0,len(valid)):
   file_name = valid.iloc[images].values[0] + '.tif'
   y_valid = valid.iloc[images].values[1]
   if(y_valid == 0):
       valid_path = valid_tum
   else:
       valid_path = valid_notum
   src = os.path.join('dataset/train/' , file_name)
   dest = os.path.join(valid_path , file_name)
   shutil.copyfile(src,dest)
  
datagen = ImageDataGenerator(rescale=1.0/255,
				horizontal_flip=True,
				vertical_flip=True)
train_gen = datagen.flow_from_directory('data/train_dataset/' , 
                                        target_size = (96,96) , 
                                        batch_size = batch_size,
                                       class_mode ='categorical')
valid_gen = datagen.flow_from_directory('data/valid_dataset//' , 
                                        target_size = (96,96) , 
                                        batch_size = batch_size,
                                       class_mode ='categorical')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,SeparableConv2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from keras.backend import stack

# class Capsulenet:
#     @staticmethod
#     def build(width,height,depth,classes):
#         model =  Sequential()
        
#         inputShape = (height,width,depth)
        
#         model.add(SeparableConv2D(filters=256, kernel_size=(9,9),
#                          activation='relu',input_shape = inputShape))
        #Primary Capsules = 8
#         for i in range(0,8):
#             x = model.add(SeparableConv2D(filters=8, kernel_size=(9,9),
#                          activation='relu',input_shape = inputShape))
            
#             stack(x , axis=0)
        # conv1 = model.output
        
        # print(conv1.shape[1:4])
#         primarycaps = CapsLayer(conv1,num_out=32,vec_len=8,
#                                 layer_type="CONV",
#                                ks = 9,stride=2,b_s=32)
        # model.summary()
#         return model
        

# basemodel = Capsulenet.build(96,96,3,2) 

#main_model = Sequential()
#main_model.add(Conv2D(filters = 32,kernel_size=(5,5),
#                              activation='relu',input_shape=(96,96,3)))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 32, kernel_size = (3,3), 
#                          activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 32, kernel_size = (3,3), 
#                          activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.2))
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 64, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.20))
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(Conv2D(filters = 128, kernel_size = (3,3), 
#                 activation='relu'))
#main_model.add(BatchNormalization())
#main_model.add(MaxPool2D(pool_size=(2, 2)))
#main_model.add(Dropout(0.25))

# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(SeparableConv2D(filters = 512, kernel_size = (3,3), 
#                  activation='relu'))
# main_model.add(MaxPool2D(pool_size=(2, 2)))
# main_model.add(Dropout(0.25))

#main_model.add(Flatten())
#main_model.add(Dense(units = 500, activation = 'relu'))
#main_model.add(Dropout(0.2))
#FC => Output
#main_model.add(Dense(2, activation='softmax'))

#main_model.summary()

#Dense Net Architecture
from keras.models import Model
base_model = DenseNet121(include_top=False,weights='imagenet',input_shape = (96,96,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(150,activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2,activation='softmax')(x)

main_model = Model(base_model.input,predictions)

#from keras.models import load_model
#main_model = load_model('check_5_epochs_conv_aug.h5')
main_model.summary()
# In[63]:


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import Adam


# In[64]:


checkpoint_fp = "DenseNet.h5"
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='val_acc',
                             verbose=1,
                            save_best_only= True,mode='max')


# In[72]:


learning_rate = ReduceLROnPlateau(monitor='val_acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max',
                                 min_lr = 0.00001)


# In[73]:


callback = [checkpoint,learning_rate]


# In[77]:


steps_p_ep_tr =np.ceil(len(train)/batch_size)
steps_p_ep_va =np.ceil(len(valid)/batch_size)


# In[78]:


main_model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'binary_crossentropy', metrics=['accuracy'])


# In[79]:


my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
                                   validation_data = valid_gen,
                                   validation_steps = steps_p_ep_va,
                                   verbose = 1,
                                   epochs = epochs,
                                   callbacks = callback)

print("---------------------------------------------------------------")
print("model keys: ",my_model.history.keys())
print("---------------------------------------------------------------")
plt.plot(my_model.history['acc'])
plt.plot(my_model.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train','valid'],loc='upper left')
plt.savefig('Dense_Net.png')
