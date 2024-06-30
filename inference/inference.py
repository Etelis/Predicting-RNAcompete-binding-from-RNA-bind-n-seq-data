import argparse
import os
import torch
from torch.utils.data import DataLoader
from model.model import CNNModel
from train.utils import load_model
from dataset.create_dataset import RBPDataset

def inference(args):
    # Load model
    model = load_model(args.model) if args.model else load_model('saved_models/cnn_model.pth')
    model.eval()

    # Create dataset
    dataset = RBPDataset(args.data_folder, selex_files=args.selex_files, rncmpt_file=args.rncmpt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            output = model(data)
            predictions.append(output.item())

    # Write output
    with open(args.ofile, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using a CNN model on RBP data')
    parser.add_argument('ofile', type=str, help='Output file to write predictions')
    parser.add_argument('rncmpt', type=str, help='Path to RNAcompete sequences file')
    parser.add_argument('selex_files', type=str, nargs='+', help='Paths to SELEX files')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--model', type=str, help='Path to the pre-trained model file')
    
    args = parser.parse_args()
    inference(args)
