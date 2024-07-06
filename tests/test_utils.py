import os
import pytest
from dataset.utils import count_kmers, calculate_kmer_ratios, compute_average_kmer_ratios, save_results

# Path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

@pytest.fixture
def setup_rbns_kmer_counts():
    """
    Fixture to count k-mers from RBNS files.
    """
    rbns_files = [f"RBP1_{i}.txt" for i in range(1, 5)]
    rbns_kmer_counts = []

    for rbns_file in rbns_files:
        file_path = os.path.join(data_folder, 'htr-selex', rbns_file)
        kmer_counts = count_kmers(file_path)
        rbns_kmer_counts.append(kmer_counts)
    
    return rbns_kmer_counts

def test_count_kmers():
    """
    Test count_kmers function with an example RBNS file.
    """
    test_file_path = os.path.join(data_folder, 'htr-selex', 'RBP1_1.txt')
    kmer_counts = count_kmers(test_file_path)

    assert isinstance(kmer_counts, dict)
    assert len(kmer_counts) > 0
    print(f"K-mer counts from {test_file_path}: {kmer_counts}")

def test_calculate_kmer_ratios(setup_rbns_kmer_counts):
    """
    Test calculate_kmer_ratios function using the first two RBNS files.
    """
    input_counts = setup_rbns_kmer_counts[0]
    reference_counts = setup_rbns_kmer_counts[1]

    ratios = calculate_kmer_ratios(reference_counts, input_counts)

    assert isinstance(ratios, dict)
    assert len(ratios) > 0
    print(f"K-mer ratios: {ratios}")

def test_compute_average_kmer_ratios(setup_rbns_kmer_counts):
    """
    Test compute_average_kmer_ratios function using RBNS k-mer ratios and RNAcompete sequences.
    """
    sequence_file_path = os.path.join(data_folder, 'RNAcompete_sequences.txt')

    low_kmer_ratios = calculate_kmer_ratios(setup_rbns_kmer_counts[0], setup_rbns_kmer_counts[0])
    mid_kmer_ratios = calculate_kmer_ratios(setup_rbns_kmer_counts[1], setup_rbns_kmer_counts[0])
    high_kmer_ratios = calculate_kmer_ratios(setup_rbns_kmer_counts[2], setup_rbns_kmer_counts[0])

    avg_low_ratios, avg_mid_ratios, avg_high_ratios = compute_average_kmer_ratios(
        sequence_file_path, low_kmer_ratios, mid_kmer_ratios, high_kmer_ratios
    )

    assert len(avg_low_ratios) > 0
    assert len(avg_mid_ratios) > 0
    assert len(avg_high_ratios) > 0
    print(f"Average low k-mer ratios: {avg_low_ratios}")
    print(f"Average mid k-mer ratios: {avg_mid_ratios}")
    print(f"Average high k-mer ratios: {avg_high_ratios}")
