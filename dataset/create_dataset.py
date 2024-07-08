import os
import random
import torch
from torch.utils.data import Dataset
from .preprocess import preprocess_data, compute_count_kmer_matrix

class RBPDataset(Dataset):
    def __init__(self, data_folder, rbp_numbers=None, selex_files=None, rncmpt_file="RNAcompete_sequences.txt"):
        self.data = []
        self.labels = []

        count_kmer_matrix = compute_count_kmer_matrix(data_folder, rncmpt_file)

        if rbp_numbers:
            for rbp_number in rbp_numbers:
                data, labels = preprocess_data(
                    data_folder, 
                    [os.path.join('htr-selex', f"RBP{rbp_number}_{i}.txt") for i in range(1, 5)], 
                    count_kmer_matrix=count_kmer_matrix, 
                    rbp_number=rbp_number
                )
                self.data.extend(data)
                self.labels.extend(labels)

        elif selex_files:
            data, _ = preprocess_data(
                data_folder, 
                selex_files, 
                count_kmer_matrix=count_kmer_matrix, 
                rncmpt_file=rncmpt_file
            )
            self.data = data
        else:
            raise ValueError("Either rbp_numbers or selex_files must be provided")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        if self.labels:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].unsqueeze(0)
        if len(self.labels) > 0:
            y = self.labels[idx]
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
