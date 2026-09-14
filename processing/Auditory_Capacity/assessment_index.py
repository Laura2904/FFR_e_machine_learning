from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from processing.Auditory_Capacity import indexes

def bartlett_test_sphericity(data):

    """
    Perform Bartlett's test for sphericity.

    It tests the null hypothesis that the correlation matrix is an identity matrix, 
    which would indicate that the variables are unrelated and unsuitable for structure detection.

    Parameters:
    data (numpy.ndarray): A 2D array where each column represents a feature.

    Returns:
    float: The chi-square statistic.
    float: The p-value of the test.
    int: The degrees of freedom.
    """
    data = np.asarray(data, dtype=float)

    n, p = data.shape

    if n <= 1:
        raise ValueError("Input data must have more than one sample.")
    
    R = np.corrcoef(data, rowvar=False)

    #Numerical projection
    R = (R + R.T) / 2

    sign, logdet = np.linalg.slogdet(R)
    
    if sign <= 0:
        raise ValueError("The correlation matrix is not positive definite.")    

    chi_square = -(n - 1 - (2 * p + 5) / 6) * logdet
    degrees_of_freedom = p * (p - 1) / 2
    p_value = chi2.sf(chi_square, degrees_of_freedom)

    return  chi_square, p_value, degrees_of_freedom


def correlation_test(ACI1,ACI2,GACI,data):
    """
    Perform correlation tests between the auditory capacity indexes and the features.

    """

    # Compute correlation coefficients
    corr_ACI1 = np.corrcoef(ACI1, data, rowvar=False)[0, 1:]
    corr_ACI2 = np.corrcoef(ACI2, data, rowvar=False)[0, 1:]
    corr_GACI = np.corrcoef(GACI, data, rowvar=False)[0, 1:]

    return corr_ACI1, corr_ACI2, corr_GACI 

def align_component_signs(reference_components,components):
    """
    Align PCA component signs with respect to a reference PCA.

    PCA eigenvectors are defined up to a sign:
        v and -v represent the same component.

    The sign is flipped when the scalar product with the
    corresponding reference component is negative.
    """

    aligned = components.copy()

    n_components = min(reference_components.shape[1],components.shape[1])

    for i in range(n_components):

        similarity = np.dot(reference_components[:, i],aligned[:, i])

        if similarity < 0:
            aligned[:, i] *= -1

    return aligned


def reliability_test(data):
    """
    Perform reliability tests between the auditory capacity indexes built with the whole dataset and the indexes 
    built with a subset of the data.

    """

    data = np.asarray(data,dtype=float)
    features = data[:,:-1]
    labels = data[:, -1]

    #reference PCA
    reference= indexes.principal_component_analysis(features,required_variance=0.90,normalization=True)

    (_,reference_scores,reference_components,pc_reference,_,reference_weights,_,_,_) = reference

    if pc_reference < 2:
        raise ValueError("The number of principal components is less than 2. Reliability test cannot be performed.")
    
    # Raw reference indexes
    ACI1_reference = reference_scores[:, 0]
    ACI2_reference = reference_scores[:, 1]
    GACI_reference = (reference_scores @ reference_weights)

    #5-cross validation
    k_fold= 5
    cv = StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=42)
    pc_k = np.zeros(k_fold)

    # Arrays containing the test indexes for all subjects
    ACI1_test = np.zeros(len(features))
    ACI2_test = np.zeros(len(features))
    GACI_test = np.zeros(len(features))

    for i, (idxTrain, idxTest) in enumerate(cv.split(features, labels)):

        xtrain_fold = features[idxTrain]
        xtest_fold = features[idxTest]

        _, _, eigenvectors_train,pc_train,_, weights_train,_ ,mean_train,std_train= indexes.principal_component_analysis(xtrain_fold,required_variance=0.90,
            normalization=True)

        pc_k[i] = pc_train

        if pc_train < 2:
            raise ValueError(f"The number of principal components is less than 2 in fold {i+1}. Reliability test cannot be performed.")
 
        xtest_fold_std = (xtest_fold - mean_train) / std_train

        # Align component signs with reference PCA
        n_align = min(pc_reference, pc_train)

        components_train = eigenvectors_train.copy()

        components_train[:, :n_align] = align_component_signs(reference_components[:, :n_align],eigenvectors_train[:, :n_align])

        # Project test data onto the aligned components
        ACI = xtest_fold_std @ components_train[:, :pc_train]
        ACI1_test[idxTest] = ACI[:,0]
        ACI2_test[idxTest] = ACI[:,1]
        GACI_test[idxTest] = ACI @ weights_train

    pc_mean = np.mean(pc_k)
    pc_std = np.std(pc_k, ddof=1)

    corr_ACI1 = np.corrcoef(ACI1_reference, ACI1_test)[0, 1]
    corr_ACI2 = np.corrcoef(ACI2_reference, ACI2_test)[0, 1]
    corr_GACI = np.corrcoef(GACI_reference, GACI_test)[0, 1]

    return  pc_mean, pc_std, corr_ACI1, corr_ACI2, corr_GACI,ACI1_test,ACI2_test,GACI_test 


def plot_reliability(index, index_test, correlation, xlabel, title):
    
    plt.figure(figsize=(7, 6))

    plt.scatter(
        index,
        index_test,
        s=40
    )
    min_value = min(
        np.min(index),
        np.min(index_test)
    )

    max_value = max(
        np.max(index),
        np.max(index_test)
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linewidth=2
    )

    plt.xlabel(xlabel)
    plt.ylabel("Index cross validation")

    plt.title(
        f"{title} (r = {correlation:.3f})"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()



  