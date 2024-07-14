import os
import numpy as np

def calculate_pearson_correlations(predicted_dir, true_dir):
    correlations = {}

    # List of files in the predicted directory
    predicted_files = [f for f in os.listdir(predicted_dir) if f.endswith('.txt')]

    for predicted_file in predicted_files:
        predicted_path = os.path.join(predicted_dir, predicted_file)
        true_path = os.path.join(true_dir, predicted_file)

        if os.path.exists(true_path):
            try:
                # Load data from files
                predicted_data = np.loadtxt(predicted_path)
                true_data = np.loadtxt(true_path)

                # Calculate Pearson correlation
                correlation = np.corrcoef(predicted_data, true_data)[0, 1]
                correlations[predicted_file] = correlation
            except Exception as e:
                print(f"Error calculating correlation for {predicted_file}: {e}")
                correlations[predicted_file] = None
        else:
            print(f"True file {predicted_file} not found in {true_dir}")
            correlations[predicted_file] = None

    return correlations

def print_correlations(correlations):
    total_corr = 0
    valid_correlations = 0

    for file_name, correlation in correlations.items():
        if correlation is not None:
            print(f"Pearson correlation for {file_name}: {correlation:.4f}")
            total_corr += correlation
            valid_correlations += 1
        else:
            print(f"Failed to calculate correlation for {file_name}")

    if valid_correlations > 0:
        average_correlation = total_corr / valid_correlations
        print(f"Average correlation: {average_correlation:.4f}")
    else:
        print("No valid correlations to calculate the average.")
