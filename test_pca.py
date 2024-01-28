import numpy as np
from pca_calcualtion import generate_random_data, calculate_pca

def test_generated_data_size():
    # Generate random data
    random_data = generate_random_data()

    # Calculate PCA
    pca_result, _, _ = calculate_pca(random_data)

    # Assert that the number of features is not greater than the number of samples
    assert random_data.shape[1] <= random_data.shape[0], "Number of features is greater than the number of samples"