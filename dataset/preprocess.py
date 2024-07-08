import os
import numpy as np
from scipy.sparse import csr_matrix
from .utils import count_kmers_from_file, calculate_kmer_ratios, compute_average_kmer_ratios, read_rbp_intensity_file, generate_count_kmer_matrix

def compute_count_kmer_matrix(data_folder, rncmpt_file, KMER=6):
    """
    Computes the binary k-mer matrix for the RNAcompete sequences.

    Parameters:
    data_folder (str): Path to the data folder.
    rncmpt_file (str): Path to the RNAcompete sequences file.
    KMER (int): Length of the k-mers to count.

    Returns:
    csr_matrix: Count k-mer matrix.
    """
    print(f"Computing binary k-mer matrix for {rncmpt_file}...")
    count_kmer_matrix = generate_count_kmer_matrix(os.path.join(data_folder, rncmpt_file), KMER)
    print("Count k-mer matrix computed.")
    return count_kmer_matrix

def preprocess_data(data_folder, selex_files, count_kmer_matrix=None, rncmpt_file=None, rbp_number=None, KMER=6):
    """
    Preprocess data to compute k-mer ratios and average k-mer ratios for SELEX files and RNAcompete sequences.
    
    Parameters:
    data_folder (str): Path to the data folder.
    selex_files (list): List of SELEX file names.
    binary_kmer_matrix (csr_matrix): Precomputed binary k-mer matrix.
    rncmpt_file (str): Path to the RNAcompete sequences file (used if binary_kmer_matrix is not provided).
    rbp_number (int): RBP number for labeling (optional).
    KMER (int): Length of the k-mers to count.

    Returns:
    tuple: Tuple containing a numpy array of average k-mer ratios (low, mid, high) and a numpy array of labels (if provided).
    """
    rbp_kmer_counts = []
    for selex_file in selex_files:
        file_path = os.path.join(data_folder, selex_file)
        print(f"Counting k-mers for {file_path}...")
        kmer_counts = count_kmers_from_file(file_path, KMER)
        rbp_kmer_counts.append(kmer_counts)
        print(f"K-mer counts for {file_path} completed.")
    
    input_kmer_counts = rbp_kmer_counts[0]
    cycle_kmer_counts = rbp_kmer_counts[1:]
    
    kmer_ratios = []
    for i, cycle_kmer_count in enumerate(cycle_kmer_counts):
        print(f"Calculating k-mer ratios for cycle {i + 1}...")
        kmer_ratios.append(calculate_kmer_ratios(input_kmer_counts, cycle_kmer_count))
        print(f"K-mer ratios for cycle {i + 1} completed.")
    
    if count_kmer_matrix is None:
        if rncmpt_file is None:
            raise ValueError("Either binary_kmer_matrix or rncmpt_file must be provided.")
        count_kmer_matrix = count_kmer_matrix(data_folder, rncmpt_file, KMER)

    print("Computing average k-mer ratios...")
    result_matrix = compute_average_kmer_ratios(count_kmer_matrix, kmer_ratios[0], kmer_ratios[1], kmer_ratios[2])
    print("Average k-mer ratios computed.")
    
    labels = np.array([])
    if rbp_number:
        print(f"Reading RBP intensity file for RBP{rbp_number}...")
        labels = np.array(read_rbp_intensity_file(os.path.join(data_folder, 'RNAcompete_intensities', f'RBP{rbp_number}.txt')))
        print(f"RBP intensity file for RBP{rbp_number} read.")

    return result_matrix, labels