# NORI_feature_extraction

# Feature Extraction from NoRI Microscopy using CNNs

This repository contains tools for extracting image features from whole tissue samples imaged using **normalised stimulated Raman spectroscopy (NoRI) microscopy**, using convolutional neural networks (CNNs). This method was developed as part of ongoing research at **Marc Kirschner's lab at Harvard Medical School**.

## 🚀 **Project Description**
Traditional histopathology relies on qualitative and often subjective assessments. This project utilizes **NoRI microscopy** in combination with **convolutional neural networks (CNNs)** to extract **quantitative features** related to protein and lipid distributions at sub-cellular resolution. These tools enable high-throughput, objective analysis of tissue in both healthy and diseased states.


### 🧬 **Unique Technology**
The NoRI microscopy platform measures the distribution of proteins and lipids at the cellular level. Machine-learning models provide comparisons between:
- Young vs. old cells.
- Healthy vs. diseased tissues.
- Male vs. Female samples and so on.

These insights support decision-making to **treat, protect, and extend the functional lifespan** of cells.

### 🌟 **Goals and Impact**
The project aims to:
- Extend the **physiological health span** of tissues.
- Advance understanding of tissue aging and disease progression.
- Develop new diagnostic tools and treatments for age-related conditions.

---

## 📁 **Data Samples**
Explore sample datasets from the project on [Kaggle](https://www.kaggle.com/competitions/kidney-segmentation-of-novel-microscopy-images/overview).

---

## 📁 **Repository Structure**

The repository includes a `work_directory` with the following structure:

```
work_directory/
├── train_directory/       # Directory for training-related data
│   ├── data/              # Training input data
│   ├── outputs/            # Trained models storage
│   └── train_config.json  # Configuration file for training
└── inference_directory/   # Directory for inference-related data
    ├── outputs/              # Inference input data
    └── inference_config.json # Configuration file for inference
```

---

## 🛠 **Getting Started**

### 1. **Training models**

Train neural network models for extracting features from NoRI microscopy images.

1. Clone the repository:
   ```bash
   git clone https://github.com/MDyakova/NORI_feature_extraction.git
   cd NORI_feature_extraction
   ```

2. Launch training process
   ```bash
   python ./train_models/train.py
   ```

Ensure your input data is placed in `work_directory/train_directory/data`, and the configuration file `train_config.json` is updated with necessary details.

Your data directory should follow this format:
```
data/
├── <project_name>/       # Directory for training-related data
│   ├── <class name 1>/    # Training input data for class 1 with tif files (necessarily)
│   ├── <class name 2>/    # Training input data for class 2 with tif files (necessarily)
│   └── <class name ...>/   # Configuration file for training (optional)
```

### 2. **Extract features**

Train neural network models for segmenting objects in NoRI microscopy images.

1. Launch inference process
   ```bash
   python ./inference_models/feature_extraction.py 
   ```

Make sure to update your configuration file, `inference_config.json`, with all the necessary details.

2. Your output directory `work_directory/inference_directory/outputs` will follow this format:

```
outputs/
├── <project_name>/  # Directory for inference-related data
│   ├── heatmaps/    # heatmap images displaying highlighted features related to the data.
│   ├── heatmap_values/    # statistics of heatmap values for various segmented objects.
│   └── umap/   # datasets for UMAP plots
│   └── probability_maps/   # probability maps of models for each sample.
```

3. Launch jupyter notebook to see all results
   ```bash
   python -m notebook 
   ```
   Open `inference_models/interactive_tool.ipynb`
   Follow instructions.

## 🛠 **Configuration Details**

### **Training Configuration**
The `train_config.json` file provides necessary information for running the training process.

#### Example: `train_config.json`
```json
{
    "data_information": {
        "nori_images" : "work_directory/train_directory/data/{your_project_folder}",
        "layers" : {"protein":0, "lipid":1},
        "sample_separator": "_"

    },
    "output_information": {
        "output_folder" : "work_directory/train_directory/outputs/{output_folder}",
        "task_name": "{your_task_name}"

    },
    "models_settings": {
        "tile_size":320,
        "val_group_number":-1,
        "epoch_number":20,
        "batch_size":32
    }

}
```

#### Parameter Details:
- **`data_information`**
  - `nori_images`: Path to your NoRI sample files for training. Replace `{your_project_folder}` with the location of your input data.
  - `layers`: The layers in the NoRI TIFF file corresponding to protein and lipid data.
  - `sample_separator`: The delimiter used for maps within a single sample.

- **`output_information`**
  - `output_folder`: Path where training results will be saved. Replace `{output_folder}` with the desired output directory.
  - `task_name`: Your task name. Replace `{task_name}` with the desired name.

- **`models_settings`**
  - `tile_size`: Size of image crops used during training (default: `320`).
  - `val_group_number`: The number of groups selected for cross-validation (use "-1" to include all available groups) (default: `-1`)
  - `epoch_number`: Number of training epoch (default: `20`).
  - `batch_size`: Number of images in one training batch (default: `32`).

---

### **Inference Configuration**
The `inference_config.json` file provides necessary information for training the models.

#### Example: `inference_config.json`
```json
{
    "data_information": {
        "data" : "work_directory/train_directory/outputs/{train_output_folder}",
        "task_name": "{your_task_name}"

    },
    "output_information": {
        "output_folder" : "work_directory/inference_directory/outputs/{output_folder}"

    },
    "models_settings": {
        "data_type": "tiles",
        "tile_size" : 320,
        "umap_type" : "train"
    }

}
```

#### Parameter Details:
- **`data_information`**
  - `data`: Path to your train output_folder. It should match the configuration used in the training settings.
  - `task_name`: Train task name. It should match the configuration used in the training settings.

- **`output_information`**
  - `output_folder`: Path where inference results will be saved. Replace `{output_folder}` with the desired output directory.

- **`models_settings`**
  - `tile_size`: Size of image crops used during training (default: `320`).
  - `data_type`: Data format used during training (default: `tiles`)
  - `umap_type`: Dataset type for training UMAP models (default: `train`).

#### Umap types: 
  - `train`: For training the model, the training dataset was used, and inferences were made using the validation dataset.
  - `val`: For training the model, the validation dataset was used, and inferences were made using the validation dataset.
  - `all`: For training the model, the all dataset was used, and inferences were made using the validation dataset.
---


### 📫 **Contact**
For questions or contributions, please contact:
**Mariia Diakova**
- GitHub: [MDyakova](https://github.com/MDyakova)
- email: m.dyakova.ml@gmail.com

---

Let me know if further edits are required!
