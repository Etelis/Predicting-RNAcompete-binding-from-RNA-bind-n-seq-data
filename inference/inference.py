import argparse
import os
import torch
from torch.utils.data import DataLoader
from model.model import CNNModel
from train.utils import load_model
from dataset.create_dataset import RBPDataset

def inference(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.model:
        model, model_params = load_model(args.model)
    else:
        model, model_params = load_model('cnn_model.pth')
    model.to(device)
    model.eval()

    # Create dataset
    dataset = RBPDataset('', selex_files=args.selex_files, rncmpt_file=args.rncmpt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.append(round(output.item(), 3))
            if args.verbose:
                print(f"Input: {data.cpu().numpy()}, Prediction: {output.item()}")

    # Write predictions to output file
    with open(args.ofile, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions written to {args.ofile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using a CNN model on RBP data')
    parser.add_argument('ofile', type=str, help='Output file to write predictions')
    parser.add_argument('rncmpt', type=str, help='Path to RNAcompete sequences file')
    parser.add_argument('selex_files', type=str, nargs='+', help='Paths to SELEX files')
    parser.add_argument('--model', type=str, help='Path to the pre-trained model file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()
    if args.verbose:
        print(f"Running inference with the following arguments: {args}")
    inference(args)
