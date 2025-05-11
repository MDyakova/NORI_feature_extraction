"""
Functions for feature extraction.py
"""

import os
import numpy as np
from tifffile import TiffFile
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def transform_image(data_type, tile_size):
    """
    Transform images to data format
    """
    if data_type == 'tiles':
        # Image processing
        transform = transforms.Compose([
            transforms.Resize(tile_size),
            transforms.CenterCrop(tile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    elif (data_type == 'tubules_with_context') | (data_type == 'tubules'):
        # Image processing
        transform = transforms.Compose([
            transforms.Resize(tile_size),
            transforms.CenterCrop(tile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    return transform

def make_folders(group, folder_name, class_name_real):
    """
    Make folders for output data
    """
    for layer in [2, 3, 4]:
        path_save = os.path.join(output_folder,
                                task_name,
                                'heatmaps',
                                folder_name,
                                class_name_real,
                                'layer_' + str(layer))
        # shutil.rmtree(path_save, ignore_errors=True)
        os.makedirs(os.path.join(path_save, 'full_color'), exist_ok=True)
        os.makedirs(os.path.join(path_save, 'one_color'), exist_ok=True)

    path_save_df = os.path.join(output_folder,
                                task_name,
                                'umap')
    os.makedirs(path_save_df, exist_ok=True)

    path_save_df = os.path.join(output_folder,
                                task_name,
                                'heatmap_values')
    os.makedirs(os.path.join(path_save_df), exist_ok=True)

    path_save = os.path.join(output_folder,
                                task_name,
                            'probability_maps',
                            folder_name)
    os.makedirs(path_save, exist_ok=True)

def make_umap(embeddings, umap_save_path, umap_type):
    embeddings_df = pd.DataFrame(embeddings, columns=('group', 'model_name', 'class_name', 'set_type',
                                      'image_name', 'image_path',
                                      'heatmap_path_full', 'heatmap_path_one',
                                      'pred_class_real', 'prob_class', 'embeddings'))

    if umap_type == 'train':
        train_embeddings = np.array([emb for emb in embeddings_df[embeddings_df['set_type']=='train']['embeddings'].values])
    elif umap_type == 'val':
        train_embeddings = np.array([emb for emb in embeddings_df[embeddings_df['set_type']=='val']['embeddings'].values])
    else:
        train_embeddings = np.array([emb for emb in embeddings_df['embeddings'].values])


    val_df = embeddings_df[embeddings_df['set_type']=='val']
    val_df.drop(columns=['embeddings'], inplace=True)

    val_embeddings = np.array([emb for emb in embeddings_df[embeddings_df['set_type']=='val']['embeddings'].values])
    
    reducer = umap.UMAP(random_state=42)
    umap_embeddings = reducer.fit_transform(train_embeddings)
    umap_embeddings_val = reducer.transform(val_embeddings)
    val_df['umap1'] = umap_embeddings_val.T[0]
    val_df['umap2'] = umap_embeddings_val.T[1]

    val_df.to_csv(umap_save_path, index=False)