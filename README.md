# Colon Polyp Detection using UNet

This repository contains a deep learning model based on the UNet architecture for predicting the presence of polyps in colon images. The model aims to assist in the early detection and diagnosis of colorectal cancer by accurately segmenting polyps from colonoscopy images.

## References
Paper: Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Re
## Introduction

Colorectal cancer is one of the most common types of cancer worldwide. Early detection and removal of polyps during colonoscopy can prevent the development of colorectal cancer. This project leverages the UNet architecture to create a model capable of segmenting polyps in colonoscopy images, aiding in early diagnosis and treatment.

## Usage

### Inference

To run inference using the pre-trained model:

```bash
python inference.py --input_path path/to/image.jpg --output_path path/to/output.jpg --model_path path/to/pretrained_model.h5
```

### Training

To train the model from scratch:

1. Prepare your dataset (see [Dataset](#dataset) section for details).
2. Run the training script:

    ```bash
    python train.py --data_dir path/to/dataset --epochs 50 --batch_size 16 --learning_rate 0.001
    ```

## Dataset

The dataset should be organized in the following structure:

```
dataset/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── mask1.png
    │       ├── mask2.png
    │       └── ...
    ├── val/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── mask1.png
    │       ├── mask2.png
    │       └── ...
```

- `images/`: Contains the input colonoscopy images.
- `masks/`: Contains the corresponding binary masks indicating the presence of polyps.

## Training

The `train.py` script supports various options to customize the training process. Here is an example of how to use it:

```bash
python train.py --data_dir path/to/dataset --epochs 50 --batch_size 16 --learning_rate 0.001
```

### Arguments

- `--data_dir`: Path to the dataset directory.
- `--epochs`: Number of training epochs.
- `--batch_size`: Size of the batches used during training.
- `--learning_rate`: Learning rate for the optimizer.

## Evaluation

To evaluate the model on the validation set:

```bash
python evaluate.py --data_dir path/to/dataset --model_path path/to/trained_model.h5
```

### Arguments

- `--data_dir`: Path to the dataset directory.
- `--model_path`: Path to the trained model file.

## Results

The results of the model, including accuracy, precision, recall, and F1-score, will be displayed after the evaluation. Example segmented images will also be saved to the specified output directory.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Make sure to follow the project's code style and include appropriate tests.
