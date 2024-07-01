import time
import random
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

def measure_execution_time(func):
    """
    Decorator to measure the execution time of functions.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def random_nucleotide_replacement(seq, random_choice):
    """
    Replace 'N' with a random nucleotide in the sequence.
    
    Parameters:
    seq (str): The DNA sequence.
    random_choice (list): List of nucleotides to choose from.

    Returns:
    str: The sequence with 'N' replaced.
    """
    return ''.join(random.choice(random_choice) if char == 'N' else char for char in seq)

def count_kmers(file_path, file_limit=500000, KMER=6):
    """
    Reads a file and counts k-mers, replacing 'N' with a random nucleotide.
    
    Parameters:
    file_path (str): Path to the input file.
    file_limit (int): Maximum number of lines to read.

    Returns:
    Counter: Dictionary with k-mer counts.
    """
    print(f"Reading file: {file_path}")
    kmer_counts = Counter()
    random_choice = ['A', 'C', 'T', 'G']
    
    with open(file_path, 'r') as file:
        for count, line in enumerate(file):
            if file_limit and count >= file_limit:
                break
            seq, freq = line.strip().split(',')
            freq = int(freq)
            seq = random_nucleotide_replacement(seq, random_choice)
            
            for i in range(len(seq) - KMER + 1):
                kmer = seq[i: i + KMER]
                kmer_counts[kmer] += freq
    
    return kmer_counts

def calculate_kmer_ratios(reference_kmer_counts, target_kmer_counts):
    """
    Calculate the ratio of k-mer counts between reference and target dictionaries.
    
    Parameters:
    reference_kmer_counts (Counter): Reference k-mer counts.
    target_kmer_counts (Counter): Target k-mer counts.

    Returns:
    dict: Dictionary with k-mer ratios.
    """
    return {kmer: reference_kmer_counts.get(kmer, 0) / target_kmer_counts.get(kmer, 1) for kmer in target_kmer_counts}

def compute_average_kmer_ratios(sequence_file_path, low_kmer_ratios, mid_kmer_ratios, high_kmer_ratios, KMER=6):
    """
    Reads sequences from a file and calculates average k-mer ratios for each sequence.
    
    Parameters:
    sequence_file_path (str): Path to the sequence file.
    low_kmer_ratios (dict): Low concentration k-mer ratios.
    mid_kmer_ratios (dict): Mid concentration k-mer ratios.
    high_kmer_ratios (dict): High concentration k-mer ratios.

    Returns:
    tuple: A tuple of three lists containing the average ratios for each sequence.
    """
    avg_low_ratios, avg_mid_ratios, avg_high_ratios = [], [], []
    
    def process_sequence(sequence):
        sequence = sequence.strip().replace('U', 'T')
        kmer_count = max(len(sequence) - KMER + 1, 1)  # Ensure count is at least 1 to avoid division by zero
        sum_low = sum_mid = sum_high = 0

        for i in range(len(sequence) - KMER + 1):
            kmer = sequence[i: i + KMER]
            sum_low += low_kmer_ratios.get(kmer, 0.0)
            sum_mid += mid_kmer_ratios.get(kmer, 0.0)
            sum_high += high_kmer_ratios.get(kmer, 0.0)

        return (sum_low / kmer_count, sum_mid / kmer_count, sum_high / kmer_count)
    
    with open(sequence_file_path, 'r') as file:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_sequence, file))
    
    avg_low_ratios, avg_mid_ratios, avg_high_ratios = zip(*results)
    
    return list(avg_low_ratios), list(avg_mid_ratios), list(avg_high_ratios)

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