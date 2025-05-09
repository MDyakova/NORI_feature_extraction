"""
Script to train model
"""

# import libraries
import os
import json
from dataset_utilities import (get_max_value, 
                                save_tiles, 
                                create_dataset_folder, 
                                remove_dataset_folder,
                                make_validation_sets,
                                copy_dataset)
from model_utilities import train_model

if __name__ == "__main__":
    # Load config
    with open(
        os.path.join(os.path.join("work_directory", "train_directory", "train_config.json")),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    # Sample's info
    images_directory = config["data_information"]["nori_images"]
    layers = config["data_information"]["layers"]
    separator = config["data_information"]["sample_separator"]
    output_directory = config["output_information"]["output_folder"]
    task_name = config["output_information"]["task_name"]
    tile_size = config["models_settings"]["tile_size"]
    val_group_number = config["models_settings"]["val_group_number"]
    epoch_number = config["models_settings"]["epoch_number"]
    batch_size = config["models_settings"]["batch_size"]

    # Create train directories
    os.makedirs(output_directory, exist_ok=True)

    # get max value for data normalisation
    max_value = get_max_value(images_directory, layers)

    # Save tiles of all samples for training
    (class_samples, 
    samples_number, 
    class_samples_rev) = save_tiles(images_directory, 
                                    layers, 
                                    output_directory, 
                                    task_name, 
                                    tile_size, 
                                    separator,
                                    max_value)

    # Generate validation samples list
    val_samples = make_validation_sets(class_samples, val_group_number)

    # Train models
    for step, version in range(val_group_number):
        model_name = f'{task_name}_models_v' + str(version)
        val_samples_step = val_samples[step]

        # make directory for train and val datasets
        create_dataset_folder(output_directory, 
                                task_name,
                                class_samples)

        # copy tiles to temporary train folder
        copy_dataset(output_directory, 
                    task_name,
                    class_samples_rev,
                    samples_number,
                    model_name, 
                    val_samples_step)

        # train model
        train_model(output_directory, 
                    model_name, 
                    val_samples_step, 
                    task_name, 
                    tile_size,
                    epoch_number,
                    batch_size)

        # remove temporary folder
        remove_dataset_folder(output_directory, 
                            task_name)