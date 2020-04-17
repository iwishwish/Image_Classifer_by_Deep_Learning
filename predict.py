#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import FloatTensor

import argparse

from utility import Process_argumnets, load_checkpoint, process_image, load_cat_to_name


def predict( image, checkpoint = "", top_k = 1, gpu = False,\
            category_names = 'cat_to_name.json' ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    if checkpoint == "":
        print ("Check_point save path is not selected.")
        return
    else:
        try:
            predict_model = load_checkpoint( checkpoint )
        except:
            print ("Error occurs when loading model!")
            return             
               
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if gpu:
        if device.type != 'cuda':
            print ( "Please enable GPU mode." )
            return
        
    predict_model.to( device )
    predict_model.eval()
    
    flower_image = Image.open( image )
    image_processed = process_image( flower_image )
    image_processed = torch.from_numpy(image_processed)
    image_processed.resize_((1, 3, 224, 224))
    image_processed = image_processed.to(device).float()
    
    cat_to_name = load_cat_to_name( category_names )
    if cat_to_name is None:
        print ("Cannot find category_names file!")
        return 

    with torch.no_grad():
        outputs = predict_model.forward(image_processed)
        probs, indexes  = torch.topk(outputs.data, top_k)
        probs = np.exp(probs).numpy().ravel()
        
        idx_to_class = { value:key for key,value in predict_model.class_to_idx.items() }
        
        classes =  [idx_to_class[int(idx)] for idx in list(indexes.cpu().numpy().ravel())]
        flowers = [ cat_to_name[class_idx] for class_idx in classes ]
    
    for index in range(0, top_k) :
        print ( "flower name:{} probability:{}".format( flowers[index], probs[index] ))
    
    return probs, flowers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability. ')
    parser.add_argument('image', action = 'store',  type=str, 
                    help="Input image file's path, including file name")
    parser.add_argument('checkpoint',action='store', 
                    help="model checkpoint path, including file name")
    parser.add_argument('--top_k', dest='top_k', action='store', type=int, 
                    help="Define the number of top K most likely classes.")
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=False,  
                    help="Switch on gpu mode.")
    parser.add_argument('--category_names', dest='category_names', action="store",
                    help="Define the number of top K most likely classes.")
    
    args = Process_argumnets( parser.parse_args() )
    
    predict( **args )
    
