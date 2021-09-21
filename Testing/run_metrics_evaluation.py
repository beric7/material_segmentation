# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:52:06 2021

@author: Admin
"""
from model_plus import createDeepLabv3Plus
from tqdm import tqdm
import torch
import numpy as np
import datahandler_plus
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
from metric_evaluation import plot_confusion_matrix, iterate_data


data_dir = './material_detection_data/bridge_images/Test/'
batchsize = 1

model = torch.load(f'./stored_weights_plus/var_4plus/var_4plus_weights_36.pt', map_location=torch.device('cuda'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()   # Set model to evvar_2plus_weights_28aluate mode
'''
# Create the dataloader
dataloaders = datahandler_plus.get_dataloader_sep_folder(data_dir, batch_size=batchsize)
nnnnn = dataloaders['Test']
'''
##############################################################################

iOU, f1, confm_sum = iterate_data(model, data_dir)

plot_confusion_matrix(confm_sum, target_names=['Background', 'Concrete', 'Steel', 'Metal Decking'], normalize=True, 
                      title='Confusion Matrix')
