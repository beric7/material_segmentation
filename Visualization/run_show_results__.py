# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:48:39 2020

@author: Eric Bianchi
"""

import os 
from show_results_ohev import*
from tqdm import tqdm   
import torch

# Load the trained model, you could possibly change the device from cpu to gpu if 
# you have your gpu configured.
model = torch.load(f'./stored_weights_plus/var_3plus/var_3plus_weights_30.pt', map_location=torch.device('cuda'))

# Set the model to evaluate mode
model.eval()

source_image_dir = './material_detection_data/bridge_images/Test/Images/'
destination_mask = './predicted_masks/bridges_var_3/'
destination_overlays = './combined_overlays/bridges_var_3/'
destination_ohev = './ohev/bridges_var_3/'

for image_name in tqdm(os.listdir(source_image_dir)):
    print(image_name)
    image_path = source_image_dir + image_name
    generate_images(model, image_path, image_name, destination_mask, destination_overlays, destination_ohev)