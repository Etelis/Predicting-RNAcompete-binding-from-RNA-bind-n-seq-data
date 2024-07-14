import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def encode_kmer(kmer):
    """
    Encodes a k-mer as an integer using base-4 encoding.
    
    Parameters:
    kmer (str): The k-mer sequence.
    
    Returns:
    int: Encoded integer representation of the k-mer.
    """
    encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    kmer_value = 0
    for char in kmer:
        kmer_value = kmer_value * 4 + encoding[char]
    return kmer_value

def count_kmers_from_file(sequence_file_path, KMER=6, file_limit=None):
    """
    Reads a sequence file and creates a numpy array with k-mer counts.
    
    Parameters:
    sequence_file_path (str): Path to the sequence file.
    KMER (int): Length of the k-mers to count.
    file_limit (int): Maximum number of lines to read.

    Returns:
    np.ndarray: Numpy array with k-mer counts.
    """
    max_kmers = 4 ** KMER
    kmer_counts = np.zeros(max_kmers, dtype=int)
    random_choice = ['A', 'C', 'T', 'G']
    
    with open(sequence_file_path, 'r') as file:
        for count, line in enumerate(file):
            if file_limit and count >= file_limit:
                break
            seq, freq = line.strip().split(',')
            freq = int(freq)
            seq = ''.join(random.choice(random_choice) if char == 'N' else char for char in seq)
            
            for i in range(len(seq) - KMER + 1):
                kmer = seq[i: i + KMER]
                encoded_kmer = encode_kmer(kmer)
                kmer_counts[encoded_kmer] += freq
    
    return kmer_counts

def generate_count_kmer_matrix(file_path, KMER=6):
    """
    Reads a sequence file and creates a binary matrix indicating k-mer presence.
    
    Parameters:
    file_path (str): Path to the sequence file.
    KMER (int): Length of the k-mers to count.

    Returns:
    np.ndarray: Binary matrix indicating k-mer presence for each line.
    """
    with open(file_path, 'r') as file:
        num_lines = sum(1 for line in file)
    
    max_kmers = 4 ** KMER
    binary_matrix = np.zeros((num_lines, max_kmers), dtype=int)
    
    with open(file_path, 'r') as file:
        for line_idx, line in enumerate(file):
            seq = line.strip().replace('U', 'T')
            
            for i in range(len(seq) - KMER + 1):
                kmer = seq[i: i + KMER]
                encoded_kmer = encode_kmer(kmer)
                binary_matrix[line_idx, encoded_kmer] += 1
    
    return binary_matrix

def calculate_kmer_ratios(reference_kmer_counts, target_kmer_counts):
    """
    Calculate the ratio of k-mer counts between reference and target arrays.
    
    Parameters:
    reference_kmer_counts (np.ndarray): Reference k-mer counts.
    target_kmer_counts (np.ndarray): Target k-mer counts.

    Returns:
    np.ndarray: Array with k-mer ratios.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.true_divide(reference_kmer_counts, target_kmer_counts)
        ratios[~np.isfinite(ratios)] = 0  # -inf inf NaN will be replaced by 0
    return ratios

def compute_average_kmer_ratios(binary_kmer_matrix, low_kmer_ratios, mid_kmer_ratios, high_kmer_ratios):
    """
    Computes the average k-mer ratios for each sequence and returns a matrix.
    
    Parameters:
    binary_kmer_matrix (np.ndarray): Binary matrix indicating k-mer presence for each line.
    low_kmer_ratios (np.ndarray): Low concentration k-mer ratios.
    mid_kmer_ratios (np.ndarray): Mid concentration k-mer ratios.
    high_kmer_ratios (np.ndarray): High concentration k-mer ratios.

    Returns:
    np.ndarray: A matrix with three columns containing the average ratios for each sequence.
    """
    weighted_low = np.dot(binary_kmer_matrix, low_kmer_ratios)
    weighted_mid = np.dot(binary_kmer_matrix, mid_kmer_ratios)
    weighted_high = np.dot(binary_kmer_matrix, high_kmer_ratios)
    
    result_matrix = np.vstack([weighted_low, weighted_mid, weighted_high]).T
    
    return result_matrix

def read_rbp_intensity_file(labels_file_path):
    """
    Reads the RBP intensity file and returns a list of true labels.

    Parameters:
    labels_file_path (str): Path to the RBP intensity file.

    Returns:
    list: A list of float values representing the true labels (RNA intensities).
    """
    true_labels = []
    
    with open(labels_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                true_labels.append(float(line))
    
    return true_labels

def save_results(output_path, results):
    """
    Writes results to a file.
    
    Parameters:
    output_path (str): Path to the output file.
    results (list): List of results to write.
    """
    print(f"Writing results to: {output_path}")
    with open(output_path, 'w') as file:
        file.write('\n'.join(map(str, results)) + '\n')
