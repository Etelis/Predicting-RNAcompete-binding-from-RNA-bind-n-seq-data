import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import CNNModel
from dataset.create_dataset import RBPDataset, split_dataset
from .utils import save_model, load_model, hyperparameter_optimization

def train_model(args):
    # Create dataset
    dataset = RBPDataset(args.data_folder, rbp_numbers=args.train_rbp_numbers)
    print(f"Total dataset size: {len(dataset)}")

    # Split dataset
    if args.train_val_ratio > 0:
        train_dataset, val_dataset = split_dataset(dataset, args.train_val_ratio)
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_dataset = dataset
        print(f"Training dataset size: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    for inputs, _ in train_loader:
        print(f"Batch input shape: {inputs.shape}")
        break

    # Load or initialize model
    model_params = {
        'num_conv_layers': args.num_conv_layers,
        'conv_filters': args.conv_filters,
        'kernel_sizes': args.kernel_sizes,
        'dropout_rate': args.dropout_rate,
        'fc_sizes': args.fc_sizes,
        'input_channels': 3
    }
    
    if args.load_model:
        model = load_model(args.load_model, model_params)
    else:
        model = CNNModel(**model_params)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.hyperparam_opt:
        best_params = hyperparameter_optimization(train_loader, val_loader if args.train_val_ratio > 0 else None)
        model = CNNModel(**best_params)
        optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            # Debug: Print input shape at the start of each epoch
            if i == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch input shape: {inputs.shape}")
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if args.verbose:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    save_model(model, args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN model on RBP data')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--train_rbp_numbers', type=int, nargs='+', required=True, default=list(range(1, 39)), help='List of RBP numbers to use for training')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='Ratio for splitting training and validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_model', type=str, default='cnn_model.pth', help='File name to save the trained model')
    parser.add_argument('--load_model', type=str, help='File name of the pre-trained model to load')
    parser.add_argument('--hyperparam_opt', action='store_true', help='Flag to perform hyperparameter optimization')
    parser.add_argument('--verbose', action='store_true', help='Flag to enable verbose logging during training')
    
    # Hyperparameters
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of convolutional layers')
    parser.add_argument('--conv_filters', type=int, nargs='+', default=[32, 64], help='Number of filters in each convolutional layer')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 3], help='Kernel sizes for each convolutional layer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the dropout layers')
    parser.add_argument('--fc_sizes', type=int, nargs='+', default=[64, 32], help='Sizes of the fully connected layers')
    
    args = parser.parse_args()
    train_model(args)
