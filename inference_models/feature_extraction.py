"""
Script to extract features from models
"""

# import libraries
import os
import json
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
import torch.nn.functional as F
from scipy.ndimage import zoom
import umap
from utilities import (transform_image,
                        make_folders,
                        make_umap,
                        apply_gradcam)

if __name__ == "__main__":
    # Load config
    with open(
        os.path.join(os.path.join("work_directory", "inference_directory", "inference_config.json")),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    # Sample's info
    data_directory = config["data_information"]["data"]
    output_directory = config["output_information"]["output_folder"]
    task_name = config["data_information"]["task_name"]
    tile_size = config["models_settings"]["tile_size"]
    data_type = config["models_settings"]["data_type"]
    umap_type = config["models_settings"]["umap_type"]


    # Create train directories
    os.makedirs(output_directory, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Only tiles
    fig, ax = plt.subplots()
    models_folder = os.path.join(data_directory, task_name, 'models')
    model_list = list(filter(lambda p: ('.pth' in p) & ('_' + task_name + '.' in p),
                            os.listdir(models_folder)))

    image_folder = os.path.join(data_directory, task_name, 'tiles')
    transform = transform_image(data_type, tile_size)

    tables_folder = os.path.join(data_directory, task_name, 'tables')

    for model_name in model_list[0:]:
        folder_name = model_name.split('.')[0].replace('_' + task_name, '')
        samples_df = pd.read_csv(os.path.join(tables_folder, folder_name + '.csv'))
        embeddings = []
        prob_masks_dict = {}
        images_dict = {}

        class_names = np.sort(np.unique(samples_df['folder']))
        num_classes = len(class_names)
        predict_classes_rev = {}
        for step, class_name in enumerate(class_names):
            predict_classes_rev[step] = class_name
       
        # Initialize the model
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Move the model to the GPU if available
        model = model.to(device)

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(os.path.join(models_folder,
                                                    model_name),
                                        map_location=torch.device('cpu')
                                        ))

        # Dictionary to store outputs
        layer_outputs = {}

        # Define a hook function
        def get_hook_fn(layer_name):
            def hook_fn(module, input, output):
                layer_outputs[layer_name] = output.detach()  # Store the output for each layer separately
            return hook_fn


        # Create a mapping from model layer indices to actual layers
        layer_mapping = {
            # 2: model.layer2,
            # 3: model.layer3,
            # 4: model.layer4,
            0: model.avgpool
        }

        # Register hooks for all layers at once
        hooks = {layer: layer_mapping[layer].register_forward_hook(get_hook_fn(layer)) for layer in layer_mapping}

        for set_type in ['train', 'val']:
            for class_name in class_names:
                sample_names = pd.unique(samples_df[(samples_df['set_type']==set_type)
                                        & (samples_df['folder']==class_name)]['sample_name'])

                make_folders(folder_name, class_name, output_directory, task_name)

                for sample_name in sample_names:
                    image_names = samples_df[(samples_df['set_type']==set_type)
                                            & (samples_df['folder']==class_name)
                                            & (samples_df['sample_name']==sample_name)]['image_name']
                    # Change
                    image_names = [i.replace(sample_name + '_', '', 1) for i in image_names]

                    for image_name in image_names:
                        image_path = os.path.join(image_folder,
                                                sample_name,
                                                image_name)
                        file_save_name = '_'.join(image_name.split('_')[:-2])
                        file_sample = class_name

                        image = Image.open(image_path)
                        image_save = image.copy()
                        image_tensor = transform(image_save).unsqueeze(0).to(device)
                        image_save_max = np.array(image_save).max(axis=2)

                        image = transform(image).unsqueeze(0)  # Add batch dimension
                        image = image.to(device)

                        # Set the model to evaluation mode
                        model.eval()

                        # Perform inference
                        with torch.no_grad():
                            output = model(image)
                            predictions = torch.argmax(output, dim=1)
                            probabilities = F.softmax(output, dim=1)
                            pred_class_name = predict_classes_rev[predictions.tolist()[0]]
                            prob_class = probabilities.max().tolist()
                            # pred_class_real = class_names[pred_class]
                            class_index = list(class_names).index(class_name)
                            class_prob = probabilities[0][class_index]
                        # Iterate over layers without if-statements in the loop
                        if set_type == 'val':
                            # Heatmaps
                            for model_layer in [2, 3, 4]:
                                pred_class = torch.argmax(output, dim=1).item()
                                heatmap = apply_gradcam(model, image_tensor, target_class=pred_class, layer=model_layer)
                                # layer_output = layer_outputs[model_layer]
                                # heatmap = np.array([layer for step, layer in enumerate(layer_output[0].cpu().numpy())])
                                # heatmap = heatmap.mean(axis=0)
                                k1 = np.array(image_save).shape[0]/heatmap.shape[0]
                                k2 = np.array(image_save).shape[1]/heatmap.shape[1]
                                heatmap_zoom = zoom(heatmap, zoom=(k1, k2))
                                heatmap_coeff = np.quantile(heatmap_zoom, 0.75)
                                masked_heatmap = np.where(heatmap_zoom >= heatmap_coeff, heatmap_zoom, np.nan)
                                heatmap_img = plt.imshow(heatmap_zoom, cmap='jet', alpha=0.4)

                                path_save_heatmaps = os.path.join(
                                                                output_directory,
                                                                task_name,
                                                                'heatmaps',
                                                                folder_name,
                                                                class_name,
                                                                'layer_' + str(model_layer))

                                # Full color heatmap overlay
                                ax.clear()  # Clear axis for reuse
                                ax.imshow(image_save.convert('L'), cmap='gray')
                                heatmap_img = ax.imshow(heatmap_zoom, cmap='jet', alpha=0.4)
                                colorbar = fig.colorbar(heatmap_img, ax=ax)  # Store colorbar reference
                                heatmap_path_full = os.path.join(path_save_heatmaps, 'full_color', image_name)
                                fig.savefig(heatmap_path_full, bbox_inches='tight')
                                colorbar.remove()  # Remove colorbar after saving
                                ax.clear()

                                # # One-color heatmap overlay (masked)
                                # ax.imshow(image_save.convert('L'), cmap='gray')
                                # masked_heatmap_img = ax.imshow(masked_heatmap, cmap='Reds', alpha=0.7)
                                # colorbar = fig.colorbar(masked_heatmap_img, ax=ax)  # Store colorbar reference
                                heatmap_path_one = os.path.join(path_save_heatmaps, 'one_color', image_name)
                                # fig.savefig(heatmap_path_one, bbox_inches='tight')
                                # colorbar.remove()  # Remove colorbar after saving
                                # ax.clear()

                            # Umap plot
                            layer_output = layer_outputs[0]
                            embeddings.append([task_name, model_name.split('.')[0], class_name, set_type,
                                                image_name, image_path, heatmap_path_full, heatmap_path_one,
                                                pred_class_name, prob_class,
                                                list(layer_output.reshape(-1).cpu().numpy())])

                        else:
                            # Umap plot
                            layer_output = layer_outputs[0]
                            embeddings.append([task_name, model_name.split('.')[0], class_name, set_type,
                                                image_name, image_path, '', '',
                                                pred_class_name, prob_class,
                                                list(layer_output.reshape(-1).cpu().numpy())])
        # Make Umap plot
        path_save_df = os.path.join(output_directory,
                                    task_name,
                                    'umap')
        umap_save_path = os.path.join(path_save_df, model_name.replace('.pth', '.csv'))
        make_umap(embeddings, umap_save_path, umap_type)
    # Close the reusable figure to release memory
    plt.close(fig)

