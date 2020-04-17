#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import FloatTensor

import argparse
import datetime

from utility import  load_data, load_cat_to_name, build_network

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def train_models( arch = 'vgg13', data_directory = 'flowers', 
                 hidden_units = 512, save_dir = './', gpu = False ,
                 epochs =3, learning_rate = 0.001, dropout = 0.5 ):
    try:
        train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx = \
        load_data( data_directory )
    except:
        print ( "Data directory is not correct.")
        return
    
    device = torch.device("cuda:0"
     if torch.cuda.is_available() else "cpu")
 
    if gpu:
        if device.type != 'cuda':
            print ( "Please enable GPU mode." )
            return

    #training
    model = build_network( arch, hidden_units, dropout )
    if model is None:
        return 
    model.to(device)
    model.train()
               
    print_every = 40
    running_loss = 0
    steps = 0
    acc_score = 0

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for e in range(epochs):
        steps = 0
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
            
               # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation( model, valid_dataloaders,
                                                     criterion, device )
                acc_score = accuracy/len(valid_dataloaders)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "steps:{}".format(steps),
                      "test_loss:{}".format(test_loss/len(valid_dataloaders)),
                      "accuracy:{}".format(acc_score))

                running_loss = 0
                model.train()
    
    #  Save model checkpoint 
    checkpoint = {'state_dict': model.classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': class_to_idx,
                  'epochs': epochs ,
                  'arch': arch,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learning_rate': learning_rate
                 }
    
    save_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
    try:
        torch.save( checkpoint, "{}/checkpoint_{}_{}_{}_{}.pth".\
                   format(save_dir, arch, hidden_units, acc_score, save_time ) )
    except:
        print ("Error occured! Checkpoint will not be saved in path: {} .\n\
        Instead it will be saved in ./checkpoint_{}_{}_{}_{}.pth".\
               format(save_dir, arch, hidden_units, acc_score, save_time))
        torch.save( checkpoint, "./checkpoint_{}_{}_{}_{}.pth".\
                   format( arch, hidden_units, acc_score, save_time ) )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train new network and \
                                     save the model as a checkpoint.")
    parser.add_argument( 'data_directory', action = 'store', \
                        help="Data directory")
    parser.add_argument('--save_dir', dest='save_dir', action='store',\
                        help="Save_directory")
    parser.add_argument('--arch', dest='arch', action='store',\
                        help="Model arch")
    parser.add_argument('--learning_rate', dest='learning_rate', action='store',\
                        type=float, help="Learning rate")
    parser.add_argument('--hidden_units', dest='hidden_units', action='store',
                        type=int, help="Model hidden units")
    parser.add_argument('--dropout', dest='dropout', action='store',
                        type=float, help="Model dropout")
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=False,  
                    help="Switch on gpu mode.")   
    parser.add_argument('--epochs', dest='epochs', action="store", type=int,  
                    help="Epochs")   
    
    args = vars( parser.parse_args() )
    drop_list = []
    for key, value in args.items():
        if value is None:
            drop_list.append( key )
    
    for elem in drop_list:
        args.pop( elem )

    train_models( **args ) 
     
           
            