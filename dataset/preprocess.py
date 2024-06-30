import os
from .utils import count_kmers, calculate_kmer_ratios, compute_average_kmer_ratios, read_rbp_intensity_file

def preprocess_data(data_folder, selex_files, rncmpt_file=None, rbp_number=None):
    """
    Preprocess data to compute k-mer ratios and average k-mer ratios for SELEX files and RNAcompete sequences.
    
    Parameters:
    data_folder (str): Path to the data folder.
    selex_files (list): List of SELEX file names.
    rncmpt_file (str): Path to the RNAcompete sequences file.
    rbp_number (int): RBP number for labeling (optional).

    Returns:
    tuple: Tuple containing a list of average k-mer ratios (low, mid, high) and labels (if provided).
    """
    rbp_kmer_counts = []
    for selex_file in selex_files:
        file_path = os.path.join(data_folder, selex_file)
        kmer_counts = count_kmers(file_path)
        rbp_kmer_counts.append(kmer_counts)
    
    input_kmer_counts = rbp_kmer_counts[0]
    cycle_kmer_counts = rbp_kmer_counts[1:]
    
    kmer_ratios = []
    for cycle_kmer_count in cycle_kmer_counts:
        kmer_ratios.append(calculate_kmer_ratios(input_kmer_counts, cycle_kmer_count))
    
    if rncmpt_file:
        avg_low_ratios, avg_mid_ratios, avg_high_ratios = compute_average_kmer_ratios(
            os.path.join(data_folder, rncmpt_file), kmer_ratios[0], kmer_ratios[1], kmer_ratios[2]
        )
    else:
        avg_low_ratios, avg_mid_ratios, avg_high_ratios = compute_average_kmer_ratios(
            os.path.join(data_folder, 'RNAcompete_sequences.txt'), kmer_ratios[0], kmer_ratios[1], kmer_ratios[2]
        )
    
    data = list(zip(avg_low_ratios, avg_mid_ratios, avg_high_ratios))
    
    labels = []
    if rbp_number:
        labels = read_rbp_intensity_file(os.path.join(data_folder, 'RNAcompete_intensities', f'RBP{rbp_number}.txt'))
    
    return data, labels
