import numpy as np

def principal_component_analysis(data, required_variance, normalization):
    """Compute Principal Component Analysis.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array where each column represents a feature.
    required_variance : float
        Fraction of variance to be explained by the retained PCs (0 < value <= 1).
    normalization : bool
        Whether to standardize the data before PCA.

    Returns
    -------
    data_new : numpy.ndarray
        Reconstructed data using the retained principal components.
    index : numpy.ndarray
        PCA scores for the retained principal components.
    eigenvectors : numpy.ndarray
        Eigenvectors of the covariance matrix.
    pc : int
        Number of retained principal components.
    variance_explained : numpy.ndarray
        Variance explained by each principal component.
    weights : numpy.ndarray
        Weights of the retained principal components.
    mean_error : float
        Sum of the discarded eigenvalues.
    mean : numpy.ndarray
        Mean of each original feature.
    std : numpy.ndarray
        Standard deviation of each original feature.
    """

    data = np.asarray(data, dtype=float)

    # Normalization
    mean = np.mean(data, axis=0)

    if normalization:
        std = np.std(data, axis=0, ddof=1)

        # Avoid division by zero for constant features
        std[std == 0] = 1

        P = (data - mean) / std

    else:
        std = np.ones(data.shape[1])
        P = data - mean

    # Covariance matrix
    sigma = np.cov(P, rowvar=False)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Deterministic sign convention
    # Force the largest loading of each PC to be positive
    for i in range(eigenvectors.shape[1]):
        max_loading = np.argmax(
            np.abs(eigenvectors[:, i])
        )

        if eigenvectors[max_loading, i] < 0:
            eigenvectors[:, i] *= -1

    # Explained variance
    total_variance = np.sum(eigenvalues)

    if total_variance <= 0:
        raise ValueError(
            "Total variance is zero. PCA cannot be performed."
        )

    variance_explained = eigenvalues / total_variance

    # Number of PCs required to reach the desired variance
    cumulative_variance = np.cumsum(variance_explained)

    pc = (
        np.argmax(cumulative_variance >= required_variance)
        + 1
    )

    # PCA scores
    new_rapp = P @ eigenvectors

    index = new_rapp[:, :pc]

    # Weights for the Global Auditory Capacity Index
    weights = (
        eigenvalues[:pc]
        / np.sum(eigenvalues[:pc])
    )

    # Dimensionality reduction
    p = np.zeros_like(new_rapp)
    p[:, :pc] = new_rapp[:, :pc]

    # Reconstruction error
    mean_error = np.sum(eigenvalues[pc:])

    # Reconstruct data
    X_std = p @ eigenvectors.T

    if normalization:
        data_new = (X_std * std) + mean
    else:
        data_new = X_std + mean

    return (data_new,index,eigenvectors,pc,variance_explained,weights,mean_error,mean,std,
    )


def normalize_index(x):
    """
    Normalize the index to a range of 0 to 100."""

    x = np.asarray(x, dtype=float)

    x_min = np.min(x)
    x_max = np.max(x)

    if np.isclose(x_max, x_min):
        return np.zeros_like(x)  # Return an array of zeros if all values are the same

    return 100 * (x - x_min) / (x_max - x_min)



def Auditory_Capacity_Index(data, required_variance=0.90):
    """
    Compute tracking, transitory and global auditory capacity indexes
    """

    normalization = True

    _, scores, _, pc, _, weights, _ ,_,_= principal_component_analysis(
        data,
        required_variance,
        normalization
    )

    #check
    if pc < 2:
        raise ValueError("The number of principal components is less than 2. Cannot compute auditory capacity indexes.")
    

    # Tracking encoding capacity
    ACI_1 = scores[:, 0]
    ACI_p_1 = normalize_index(ACI_1)

    # Transitory encoding capacity
    ACI_2 = scores[:, 1]
    ACI_p_2 = normalize_index(ACI_2)

    # Global auditory encoding capacity
    GACI = scores @ weights
    GACI_p = normalize_index(GACI)


    return ACI_p_1, ACI_p_2, GACI_p