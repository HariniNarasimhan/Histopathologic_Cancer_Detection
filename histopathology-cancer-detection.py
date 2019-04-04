
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import os
import shutil
import skimage.io as skio
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.applications import inception_v3,nasnet,mobilenet,vgg19,resnet50,xception
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from sklearn.utils import shuffle


# reading the total training labels
#"train_labels.csv" will have the labels of all the images 
file = pd.read_csv("../input/train_labels.csv")


#'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2 and 9369c7278ec8bcc6c880d99194de09fc2bd4efbe'
# these two are the images with full black, which is not needed to train the model
file =file[file['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2'] 
file =file[file['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

print("total number of images: ",file.shape)

#values_counts will give the total number of different labels 
file['label'].value_counts()

#since the dataset is biased towards 'label 0'(no tumour) we are taking equal number of data 
#from each label and then concatenating into one variable and shuffle it. 
f_0 = file[file['label'] == 0].sample(80000,random_state = 101)
f_1 = file[file['label'] == 1].sample(80000,random_state = 101)
file = pd.concat([f_0,f_1],axis=0).reset_index(drop = True)
file = shuffle(file)

file['label'].value_counts()

#storing all the labels in variable y
y = file['label']

#splitting the data for testing and training(20 and 80%) using train_test_split command imported from sklearn
#random_state will always choose the same data for every trial and stratify is to take equal number of abnormal 
#and normal data from total dataset
x_train,x_valid = train_test_split(file,test_size = 0.20,random_state= 101,stratify=y)

print(x_train.shape)
print(x_valid.shape)


x_train['label'].value_counts()


x_valid['label'].value_counts()

#function to create a folder when it doesnt exist
def create_folder(folderName):
    if not os.path.exists(folderName):
        try:
            os.makedirs(folderName)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


#creating different folders for data/train_dataset and data/valid_dataset
base_dir = 'data'
create_folder(base_dir)


train_dir = os.path.join(base_dir,'train_dataset')
create_folder(train_dir)
valid_dir = os.path.join(base_dir,'valid_dataset')
create_folder(valid_dir)


#inside train_dataset and valid_dataset folder create two more folders 0 and 1(normal and abnormal)
train_tum = os.path.join(train_dir,'0')
create_folder(train_tum)
train_notum = os.path.join(train_dir,'1')
create_folder(train_notum)

valid_tum = os.path.join(valid_dir,'0')
create_folder(valid_tum)
valid_notum = os.path.join(valid_dir,'1')
create_folder(valid_notum)


# check that the folders have been created
os.listdir('data/train_dataset//')

# Set the id as the index in df_data
file.set_index('id', inplace=True)

# Get a list of train and val images
train_list = list(x_train['id'])
val_list = list(x_valid['id'])


# Transfer the train images into 0 and 1 respectively

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = file.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = '0'
    if target == 1:
        label = '1'
    
    # source path to image
    src = os.path.join('../input/train', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the validation images into 0 and 1 respectively

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = file.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = '0'
    if target == 1:
        label = '1'
    

    # source path to image
    src = os.path.join('../input/train', fname)
    # destination path to image
    dst = os.path.join(valid_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


batch_size = 90
epochs = 10


# In[20]:


#doing data augmentation - (creating more images from availbale images by normalising all the images 
#flipping the images horizontally and vertically)
datagen = ImageDataGenerator(rescale=1.0/255,
                horizontal_flip=True,
                vertical_flip=True)

#resizing all the images to 96 x 96
train_gen = datagen.flow_from_directory('data/train_dataset/' , 
                                        target_size = (96,96) , 
                                        batch_size = batch_size,
                                       class_mode ='categorical')


# def tr_x(tr_gen):
#     for x,y in tr_gen:
#         print(x.shape)
#         yield x
# def tr_y(tr_gen):
#     for x,y in tr_gen:
#         yield y


valid_gen = datagen.flow_from_directory('data/valid_dataset/',
					target_size = (96,96),
					batch_size = batch_size,
					class_mode='categorical')

# def va_x(val_gen):
#     for x,y in val_gen:
#         yield x
# def va_y(val_gen):
#     for x,y in val_gen:
#         yield y

#to take only center 32 x 32 patch from the given image
def patches(mode):
    
    if (mode == 'valid'):
        xy = valid_gen
    elif(mode == 'train'):
        xy = train_gen
    else:
        xy = test_gen

    batches = 0
    for x,y in xy:
        s = x.shape
        print(x)
        img = x[:,32:64,32:64,:]
        img = np.resize(img,s)
        batches += 1
#         yield ([img,y],[y,img])
        yield img,y


# In[24]:


patches('valid')


# In[25]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,SeparableConv2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from keras import layers,models
from keras import initializers



#function for building the pretrained architecture
def pretrained_model(model):
    if model == 'densenet':
        base_model = DenseNet121(include_top=False,weights='imagenet',input_shape = (96,96,3))
    elif model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape = (96,96,3))
    elif model == 'mobilenet':
        base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape = (96,96,3))
    elif model == 'vgg':
        base_model = vgg19.VGG19(include_top=False,weights='imagenet',input_shape = (96,96,3))
    elif model == 'resnet':
        base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape = (96,96,3))
    elif model == 'xception':
        base_model = xception.Xception(include_top=False,weights='imagenet',input_shape = (96,96,3))
        
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(150,activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2,activation='softmax')(x)

    return models.Model(base_model.input,predictions)


main_model = pretrained_model('vgg')
main_model.summary()


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam,RMSprop

#CSV_Logger is to store all the accuracy and loss values into a csv file for every epoch 
#ModelCheckpoint is to save the best models amoung all the epochs
#Learning rate starts at 0.001 should keep on reducing at the factor of 0.1 if there is no change in validation accuracy

csv_logger = CSVLogger("result.csv",separator = ",",append=True)

checkpoint_fp = "vgg_model.h5"
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='val_acc',
                             verbose=1,
                            save_best_only= True,mode='max')

learning_rate = ReduceLROnPlateau(monitor='val_acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max',
                                 min_lr = 0.00001)

callback = [checkpoint,learning_rate,csv_logger]


# In[31]:


steps_p_ep_tr =np.ceil(len(x_train)/batch_size)
steps_p_ep_va =np.ceil(len(x_valid)/batch_size)


main_model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'binary_crossentropy', metrics=['accuracy'])

#training the model for all the images
my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
                                   validation_data = valid_gen,
                                   validation_steps = steps_p_ep_va,
                                   verbose = 1,
                                   epochs = epochs,
                                   callbacks = callback)


# to remove all the data folder create earlier
shutil.rmtree('data')


# create test_dir
test_dir = 'test_dir'
os.mkdir(test_dir)
    
# create test_images inside test_dir
test_images = os.path.join(test_dir, 'test_images')
os.mkdir(test_images)

os.listdir('test_dir/')


test_list = os.listdir('../input/test')

#moving all the test images to test folder 
for image in test_list:
    
    fname = image
    
    # source path to image
    src = os.path.join('../input/test', fname)
    # destination path to image
    dst = os.path.join(test_images, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# In[42]:


test_gen = datagen.flow_from_directory('test_dir/',target_size = (96,96),
                    batch_size = batch_size,
                    class_mode='categorical',
                    shuffle= False)
# 
def te(te_gen):
    for x,y in te_gen:
        yield ([x,y],[y,x])


# In[43]:


# make sure we are using the best epoch
#load the best weights you have stored (the best learned model from training images)
main_model.load_weights('vgg_model.h5')

#predicting your labels for test data
predictions = main_model.predict_generator(test_gen, steps=57458, verbose=1)


# In[44]:


predictions.shape

#model predicted will be of probability values but our model shoudl have either 0 or 1
# so take the position of maximum values for each data
test_preds = np.argmax(predictions,axis = 1)
test_preds.shape

#Store those values in dataframe
f_preds = pd.DataFrame(test_preds, columns=['label'])

f_preds.head()

#extracting filenames of each test data
test_filenames = test_gen.filenames

# add the filenames to the dataframe
f_preds['file_names'] = test_filenames

f_preds.head()

#function to etract only the id of all the test images
def extract_id(x):
    
    # split into a list
    a = x.split('/')
    # split into a list
    b = a[1].split('.')
    extracted_id = b[0]
    
    return extracted_id

f_preds['id'] = f_preds['file_names'].apply(extract_id)
f_preds.head()

#final submission file with two columns (1 - image id's and 2 - label predicted)
submission = pd.DataFrame({'id':f_preds['id'], 
                           'label':f_preds['label'], 
                          }).set_index('id')

submission.to_csv('submission_dense.csv', columns=['label'])

