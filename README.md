
# RBP Fully Connected Model

This repository contains scripts to train a fully connected neural network model on RNA-binding protein (RBP) data and perform inference using a trained model.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- tqdm
- Other Python packages as specified in `requirements.txt`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the fully connected model, use the `train.py` script. This script allows you to train the model on specified RBP datasets and optionally load a pre-trained model to continue training.

### Basic Usage

```bash
python train.py --data_folder <path_to_data_folder> --train_rbp_numbers <rbp_numbers> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
```

### Example Command

```bash
python train.py --data_folder ./data --train_rbp_numbers 1 2 3 4 --train_val_ratio 0.8 --epochs 20 --batch_size 32 --learning_rate 0.001 --save_model fc_model.pth --fc_sizes 128 64 32 --dropout_rate 0.5
```

### Arguments

- `--data_folder`: (Required) Path to the folder containing RBP data.
- `--train_rbp_numbers`: (Optional) List of RBP numbers to use for training. Default is 1 to 38.
- `--train_val_ratio`: (Optional) Ratio for splitting the dataset into training and validation. Default is 0.8.
- `--batch_size`: (Optional) Batch size for training. Default is 32.
- `--epochs`: (Optional) Number of training epochs. Default is 20.
- `--learning_rate`: (Optional) Learning rate for the optimizer. Default is 0.001.
- `--save_model`: (Optional) Path to save the trained model. Default is `fc_model.pth`.
- `--load_model`: (Optional) Path to a pre-trained model to load.
- `--fc_sizes`: (Optional) List of sizes for the fully connected layers. Default is `[128, 64, 32]`.
- `--dropout_rate`: (Optional) Dropout rate for regularization. Default is 0.5.
- `--verbose`: (Optional) Flag to enable verbose logging during training.

### Early Stopping and Model Saving

The training process includes early stopping based on validation loss, with the best model saved to the path specified in `--save_model`.

## Running Inference

To perform inference with a trained model, use the `inference.py` script. This script allows you to load a pre-trained model and make predictions on new RBP data.

### Basic Usage

```bash
python inference.py --ofile <output_file> --rncmpt <path_to_rncmpt_file> --selex_files <paths_to_selex_files> --model <path_to_trained_model>
```

### Example Command

```bash
python inference.py --ofile predictions.txt --rncmpt ./data/rncmpt.txt --selex_files ./data/selex1.txt ./data/selex2.txt --model fc_model.pth
```

### Arguments

- `--ofile`: (Required) Output file to write predictions.
- `--rncmpt`: (Required) Path to the RNAcompete sequences file.
- `--selex_files`: (Required) Paths to SELEX files.
- `--model`: (Required) Path to the trained model file.
- `--verbose`: (Optional) Enable verbose logging during inference.
- `--true_dir`: (Optional) Path to the directory containing true label files for correlation calculation.

### Correlation Calculation

If the `--true_dir` argument is provided, the script will calculate Pearson correlations between the predicted and true values.
