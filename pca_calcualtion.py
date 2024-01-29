"""
This piece of code calculates principal component analysis by taking n (number of samples) and m (number of features).
The data is generated randomly.

PCA fails when number of features is bigger than the number of samples. the test scenario that is going to be created
is going to test that to make sure the randomly generated data is correct.

"""
import numpy as np
from sklearn.decomposition import PCA
def generate_random_data():
    # Generate random data with 'n' samples and 'm' features
    n = int(np.random.random()*100)
    m = int(np.random.random()*100)
    data = np.random.rand(n, m)
    return data


def calculate_pca(data):
    # Center the data by subtracting the mean of each feature
    centered_data = data - np.mean(data, axis=0)

    # Create a PCA object
    pca = PCA()

    # Fit the PCA model to the centered data
    pca.fit(centered_data)

    # Access the explained variance ratios, eigenvalues, and eigenvectors
    explained_variances = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Project the data onto the principal components
    pca_result = pca.transform(centered_data)

    return pca_result, eigenvalues, eigenvectors


def main():
    # Generate random data
    random_data = generate_random_data()

    # Calculate PCA
    pca_result, eigenvalues, eigenvectors = calculate_pca(random_data)

    # Print results
    print("Original Data Shape:", random_data.shape)
    print("Principal Components Shape:", pca_result.shape)

