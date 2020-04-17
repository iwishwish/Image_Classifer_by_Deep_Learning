#!/usr/bin/env python3

import json
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import FloatTensor

from collections import OrderedDict

import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image


def load_data(  data_dir = './flowers' ):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(
                                                       [0.485, 0.456,0.406], 
                                                       [0.229, 0.224, 0.225])
                                                  ])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   [0.485, 0.456, 0.406], 
                                                   [0.229, 0.224, 0.225])
                                              ])
                                           
    # Load    the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(
        train_dir, transform=training_data_transforms )
    valid_image_dataset = datasets.ImageFolder( 
        valid_dir, transform=test_data_transforms )
    test_image_dataset = datasets.ImageFolder( 
        test_dir, transform=test_data_transforms)                                          
    class_to_idx = train_image_dataset.class_to_idx
    
    # Using the image datasets and the transforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(
        train_image_dataset, batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(
        valid_image_dataset, batch_size=32, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(
        test_image_dataset, batch_size=32, shuffle=True)
    
    return train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx

def load_cat_to_name(filepath = 'cat_to_name.json'):
    cat_to_name_dict = None
    
    try:
        with open(filepath , 'r') as f:
            cat_to_name_dict = json.load(f)
    except:
        print ( "Cannot load {}, please check!".format( filepath ) ) 
  
    return cat_to_name_dict

def build_network( arch = 'vgg16', hidden_units = 512, dropout = 0.5 ):
    #Build network
    model_intial_func = models.__dict__.get( arch )
    
    if model_intial_func is not None:
        try:
             model = model_intial_func( pretrained=True )
             #print ("model=", model)
        except:
            print ( "The selected arch is wrong or not supported, please choose another one!" ) 
            return None
    else:
        print ( "The selected arch is wrong or not supported, please choose another one!" ) 
        return None
                
    # Freeze parameters to avoid backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    try:
        try:
            in_features = list(enumerate(model.classifier.named_modules()))[0][1][1][0].in_features
        except:
            print ( "This type of arch is not supported now, please select vgg arch." )
            return None
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1)) ]))
    except Exception as e:
        print (e)
        print ( "Hidden_units should be integer!" )
        return None
    
    try:
        classifier.dropout = nn.Dropout( dropout )
    except:
        print ( "Dropout set wrong number!" )
        return None
      
    model.classifier = classifier
    #print ("model=", model)
    return model

def Process_argumnets( arguments ):
    args = vars( arguments )
    drop_list = []
    for key, value in args.items():
        if value is None:
            drop_list.append( key )
    
    for elem in drop_list:
        args.pop( elem )
    
    return args


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    ratio = image.size[0]/image.size[1] 
    
    if image.size[0] > image.size[1] :
        new_size = ( int(256*ratio)  , 256)
    else:
        new_size = ( 256, int(256/ratio))
        
    image = image.resize(new_size) 
    
    center_x = int(new_size[0]/2 )
    center_y = int(new_size[1]/2)
    left = int(center_x-224/2)
    right = left + 224
    up = int(center_y - 224/2)
    bottom = up + 224
    crop_image = image.crop( (left,up ,right , bottom) )
    
    np_image = np.array(crop_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255.0 - mean)/std
    np_image = np_image.transpose((2,0,1))
    
    return np_image
   
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return image



def load_checkpoint(filepath ):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location= "cuda:0")
    else:
        checkpoint = torch.load(filepath, map_location= "cpu")
    #print ("checkpoint:", checkpoint)
    model_new = build_network( checkpoint['arch'], checkpoint['hidden_units'], checkpoint['dropout'] )
    if model_new is None:
        return None
 
    model_new.classifier.load_state_dict(checkpoint['state_dict'])
    model_new.class_to_idx = checkpoint['class_to_idx']
    
    return model_new
    



#----------------------test code---------------------------
if __name__ == '__main__':
    #model = build_network( arch = 'vgg16', hidden_units = 77 )
    #print ( model )
    
    #model2 = build_network( arch = 'vgg16', hidden_units = '128' )
    #print ( model2 )
    
    #model3 = build_network( arch = 'vgg18', hidden_units = '128' )
    #print ( model3 )
    
    #model4 = build_network( arch = 'vgg13', hidden_units = 256 )
    #print ( model4 )
    
    #model5 = build_network( arch = 'vgg11', hidden_units = 256, dropout = 0.3 )
    #print ( model5 )
    
    #train_dataloaders, valid_dataloaders, test_dataloaders =  load_data()
    #print ( type( train_dataloaders ) )
    #print ( type( valid_dataloaders ) )
    #print ( type( test_dataloaders ) )
    
    dict1 = load_cat_to_name()
    print ( dict1 )
    dict2 = load_cat_to_name('')
    print ( dict2 )
    
    dict3 = load_cat_to_name('ds//ss')
    print ( dict3 )
    


      
           
