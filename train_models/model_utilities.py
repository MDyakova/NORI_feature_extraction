"""
Model functions for train.py
"""

import os
import numpy as np
from tifffile import TiffFile
from PIL import Image
import pandas as pd
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def random_rotation():
    """Returns a random rotation angle (0, 90, 180, or 270)."""
    return random.choice([0, 90, 180, 270])

# Training function
def train_process(model, 
                dataloaders, 
                criterion, 
                optimizer, 
                model_directory, 
                device,
                image_datasets,
                task_name,
                model_name,
                val_samples,
                num_epochs=25):
    """
    Start train process
    """
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        results_l = []
        results_p = []

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    results_l.extend(list(labels.data.cpu().numpy()))
                    results_p.extend(list(preds.cpu().numpy()))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                report = classification_report(results_l, results_p, target_names=image_datasets['train'].classes, output_dict=False)
                cf_matrix = confusion_matrix(results_l, results_p)
                f1_scores = f1_score(results_l, results_p, average='macro')
                print(f'f1: {f1_scores:.4f}')

            if phase == 'val' and f1_scores > best_f1:
                best_f1 = f1_scores
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_report = report
                best_cf_matrix = cf_matrix

    with open(os.path.join(model_directory, f'report_CNN_{task_name}.txt'), 'a') as f:
        f.write(model_name + '\n')
        f.write(' '.join(val_samples) + '\n')
        f.write(best_report + '\n')
        f.write(str(best_cf_matrix) + '\n')

    return model

def train_model(output_directory, 
                    model_name, 
                    val_samples, 
                    task_name, 
                    tile_size,
                    epoch_number,
                    batch_size):
    """
    Data transform and training model
    """

    data_dir = os.path.join(output_directory, task_name, 'dataset')
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = {
        'train': transforms.Compose([
            transforms.Resize(tile_size),
            transforms.RandomResizedCrop(tile_size),
            transforms.RandomChoice([  # 50% chance of flipping
                transforms.RandomHorizontalFlip(p=1.0),  # Always flip if chosen
                transforms.RandomVerticalFlip(p=1.0),    # Always flip if chosen
                transforms.Lambda(lambda x: x)  # No flip (identity function)
            ]),
            transforms.Lambda(lambda img: img.rotate(random_rotation())), # Apply random rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(tile_size),
            transforms.CenterCrop(tile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(root=f'{data_dir}/train', transform=transform['train']),
        'val': datasets.ImageFolder(root=f'{data_dir}/val', transform=transform['val']),
    }

    # Define dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
    }

    # Get the number of classes
    num_classes = len(image_datasets['train'].classes)

    # Load a pre-trained ResNet model and modify the final layer
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    model_directory = os.path.join(output_directory, task_name, 'models')
    os.makedirs(model_directory, exist_ok=True)

    # Train the model
    saved_model = train_process(model, 
                                dataloaders, 
                                criterion, 
                                optimizer, 
                                model_directory, 
                                device,
                                image_datasets,
                                task_name,
                                model_name,
                                val_samples,
                                num_epochs=epoch_number)

    # Save the trained model
    torch.save(saved_model.state_dict(), os.path.join(model_directory, model_name + f'_{task_name}.pth'))

    return saved_model