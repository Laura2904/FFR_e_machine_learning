#Modules to compute the PCA
import numpy as np

def principal_component_analysis(data,required_variance,normalizzation):
    """It computes the principal component analysis"""

    mean = np.mean(data,axis = 0)

    #Normalization
    if normalizzation:

        std = np.std(data, axis=0, ddof=1)
        std[std == 0] = 1
        P =(data -mean)/ std

    else:
        std = None
        P = data - mean 

    #Covariance matrix and his eigvalues and eigvectors
    sigma = np.cov(P, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance
    variance_explained = eigenvalues / np.sum(eigenvalues)
    pc = np.argmax(np.cumsum(variance_explained) >= required_variance) + 1

    new_rapp = P @ eigenvectors
    index = new_rapp[:,:pc]
    weights = eigenvalues[:pc]/np.sum(eigenvalues[:pc])

    #data dimension reduction
    p = np.zeros_like(new_rapp)
    p[:, :pc] = new_rapp[:, :pc]
    mean_error = np.sum(eigenvalues[pc:])

    #new rappresentation of data
    X_std = p @ eigenvectors.T

    if normalizzation:
        data_new =( X_std * std ) +mean
    else:
        data_new = X_std + mean

    return data_new, index, eigenvectors,pc,variance_explained, weights, mean_error, std, mean


def normalize_index(x):
        return 100 * (x - np.min(x)) / (
            np.max(x) - np.min(x) + 1e-12
        )


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