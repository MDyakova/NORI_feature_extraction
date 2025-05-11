"""
Data preprocessing functions for train.py
"""

import os
import numpy as np
from tifffile import TiffFile
from PIL import Image
import pandas as pd
import shutil
from itertools import product
import random

def image_filter(image):
    """
    Filter data outliers
    """
    all_layers = []
    for layer in range(0, 3):
        image_layer = image[layer]
        all_percentile = []
        for step_i in range(image_layer.shape[0]//256):
            for step_j in range(image_layer.shape[1]//256):
                image_layer_crop = image_layer[step_i*256:(step_i+1)*256, step_j*256:(step_j+1)*256]
                percentile_99 = np.percentile(image_layer_crop, 99)
                percentile_1 = np.percentile(image_layer_crop, 1)
                all_percentile.append(percentile_99)
        percentile_99 = np.median(all_percentile)
        image_layer = np.where((image_layer>percentile_99), percentile_99, image_layer)
        all_layers.append(image_layer)
    filtered_image = np.array(all_layers)
    return filtered_image

def read_sample(file_path, layers, max_value=None):
    """
    Read, filter and transform image
    """
    if os.path.exists(file_path):
        with TiffFile(file_path) as tif:
            image = tif.asarray()
        image_nori = []
        image_nori.append(image[layers['protein']])
        image_nori.append(image[layers['lipid']])
        if 'water' in layers:
            image_nori.append(image[layers['water']])
        else:
            image_nori.append(image[0]*0)
        image_nori = np.stack(image_nori, axis=0)
        filtered_image = image_filter(image_nori)
        if max_value is not None:
            for layer in range(0, 3):
                filtered_image[layer] = filtered_image[layer]/max_value
            transformed_image = np.array(Image.fromarray((filtered_image*255).transpose((1, 2, 0)).astype(np.uint8)))
        else:
            transformed_image = None
        return filtered_image, transformed_image

def get_max_value(data_folder, layers):
    """
    Get max values for all data layers for normalisation
    """
    all_values = []
    class_names = os.listdir(data_folder)
    for class_name in class_names:
        if os.path.isdir(os.path.join(data_folder, class_name)):
            file_names = os.listdir(os.path.join(data_folder, class_name))
            for file_name in file_names:
                if ('.tif' in file_name):
                    file_path = os.path.join(data_folder, class_name, file_name)
                    filtered_image, _ = read_sample(file_path, layers, max_value=None)
                    all_values.append(filtered_image[0].max())
                    all_values.append(filtered_image[1].max())
    return np.max(all_values)

def save_tiles(data_folder, 
                layers, 
                output_directory, 
                task_name, 
                tile_size, 
                separator, 
                max_value):
    """
    Save tiles of samples for training
    """

    save_directory = os.path.join(output_directory, task_name, 'tiles')
    shutil.rmtree(save_directory, ignore_errors=True)

    class_samples = {}
    samples_number = {}
    class_samples_rev = {}

    class_names = os.listdir(data_folder)
    for class_name in class_names:
        class_samples[class_name] = []
        samples_number[class_name] = 0
        if os.path.isdir(os.path.join(data_folder, class_name)):
            file_names = os.listdir(os.path.join(data_folder, class_name))
            for file_name in file_names:
                if ('.tif' in file_name):
                    file_path = os.path.join(data_folder, class_name, file_name)
                    sample_name = file_name.split(separator)[0]
                    file_save_directory = os.path.join(save_directory, 
                                                        sample_name)
                    if sample_name not in class_samples[class_name]:
                        class_samples[class_name].append(sample_name)
                        class_samples_rev[sample_name] = class_name
                    os.makedirs(file_save_directory, exist_ok=True)
                    _, transformed_image = read_sample(file_path, layers, max_value=max_value)
                    height, width, _ = transformed_image.shape
                    for step_i in range(0, height - 0 - tile_size, tile_size//2):
                        for step_j in range(0, width - 0 - tile_size, tile_size//2):
                            im_cell = transformed_image[step_i:step_i+tile_size, step_j:step_j+tile_size, :]
                            file_name_save = file_name.split('.')[0] + '_' + str(step_i) + '_' + str(step_j)
                            image_save = Image.fromarray(im_cell)
                            file_path_save = os.path.join(file_save_directory,
                                                          file_name_save + '.jpg')
                            image_save.save(file_path_save, format="JPEG", quality=100, optimize=False)
                            samples_number[class_name]+=1
    samples_number_min = np.min(list(samples_number.values()))
    for class_name in samples_number.keys():
        samples_number[class_name] = samples_number_min/samples_number[class_name]

    return class_samples, samples_number, class_samples_rev

def make_validation_sets(class_samples, val_group_number):
    """
    Function generate list of samples for cross validation
    """

    groups = list(product(*class_samples.values()))
    if val_group_number>0:
        val_samples = random.sample(groups, val_group_number)
    else:
        val_samples = groups
    return val_samples


def create_dataset_folder(output_directory, 
                            task_name,
                            class_samples):

    """
    Create directory for train and val datasets
    """

    dataset_directory = os.path.join(output_directory, task_name, 'dataset')
    shutil.rmtree(dataset_directory, ignore_errors=True)
    os.makedirs(dataset_directory, exist_ok=True)

    d_types = class_samples.keys()
    set_types = ['train', 'val']
    for set_type in set_types:
        for d_type in d_types:
            file_path_save = os.path.join(dataset_directory, set_type, d_type)
            if not os.path.exists(file_path_save):
                os.makedirs(file_path_save)

def remove_dataset_folder(output_directory, 
                            task_name):
    dataset_directory = os.path.join(output_directory, task_name, 'dataset')
    shutil.rmtree(dataset_directory, ignore_errors=True)

def copy_dataset(output_directory, 
                task_name,
                class_samples_rev,
                samples_number,
                model_name, 
                val_samples_step):
    dataset_info = []
    tiles_folder = os.path.join(output_directory, task_name, 'tiles')
    dataset_folder = os.path.join(output_directory, task_name, 'dataset')
    for sample in os.listdir(tiles_folder):
        folder = class_samples_rev[sample]
        balance_value = samples_number[folder]
        if os.path.isdir(os.path.join(tiles_folder, sample)):
            all_sample_images = os.listdir(os.path.join(tiles_folder, sample))
            set_type = 'val' if (sample in val_samples_step) else 'train'
            for image_name in all_sample_images:
                random_value = random.uniform(0, 1)
                if random_value<=balance_value:
                    shutil.copy(os.path.join(tiles_folder, sample, image_name),
                                os.path.join(dataset_folder, set_type, folder, sample + '_' + image_name))
                    dataset_info.append([model_name, folder, sample, set_type, sample + '_' + image_name])
    dataset_info = pd.DataFrame(dataset_info,
                                columns=['model_name', 'folder',
                                         'sample_name',
                                         'set_type', 'image_name'])
    
    tables_directory = os.path.join(output_directory, task_name, 'tables')
    os.makedirs(tables_directory, exist_ok=True)
    dataset_info.to_csv(os.path.join(tables_directory, model_name + '.csv'), index=None)
    