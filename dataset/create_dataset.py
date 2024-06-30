import os
import random
import torch
from torch.utils.data import Dataset
from .preprocess import preprocess_data

class RBPDataset(Dataset):
    def __init__(self, data_folder, rbp_numbers=None, selex_files=None, rncmpt_file=None):
        self.data = []
        self.labels = []

        if rbp_numbers:
            for rbp_number in rbp_numbers:
                data, labels = preprocess_data(data_folder, [os.path.join('htr-selex', f"RBP{rbp_number}_{i}.txt") for i in range(1, 5)], rbp_number=rbp_number)
                self.data.extend(data)
                self.labels.extend(labels)
        elif selex_files and rncmpt_file:
            data, _ = preprocess_data(data_folder, selex_files, rncmpt_file=rncmpt_file)
            self.data = data
        else:
            raise ValueError("Either rbp_numbers or selex_files and rncmpt_file must be provided")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert data to tensor and add the channel dimension (for Conv1d)
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Shape: (1, length)
        if self.labels:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x

def split_dataset(dataset, train_ratio=0.8):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    split_index = int(len(dataset) * train_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    
    return train_data, val_data
