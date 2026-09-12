import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
import indexes 
from indexes import normalize_index

def bartlett_test_sphericity(data):

    """
    Perform Bartlett's test for sphericity.

    It tests the null hypothesis that the correlation matrix is an identity matrix, 
    which would indicate that the variables are unrelated and unsuitable for structure detection.

    Parameters:
    data (numpy.ndarray): A 2D array where each column represents a group.

    Returns:
    float: The chi-square statistic.
    float: The p-value of the test.
    int: The degrees of freedom.
    """

    n, p = data.shape
    R = np.corrcoef(data, rowvar=False)
    det_R = np.linalg.det(R)

    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det_R)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi_square, degrees_of_freedom)

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


def reliability_test(ACI1,ACI2,GACI,data):
    """
    Perform reliability tests between the auditory capacity indexes built with the whole dataset and the indexes 
    built with a subset of the data.

    """
    #5-cross validation
    k_fold= 5
    cv = StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=42)
    pc_k = np.zeros(k_fold)

    features = data[:,2:-1]
    labels = data[:, -1]

    # Arrays containing the test indexes for all subjects
    ACI1_test = np.zeros(len(data))
    ACI2_test = np.zeros(len(data))
    GACI_test = np.zeros(len(data))

    for i, (idxTrain, idxTest) in enumerate(cv.split(features, labels)):

        xtrain_fold, ytrain_fold = features[idxTrain], labels[idxTrain]
        xtest_fold, ytest_fold = features[idxTest], labels[idxTest]

        _, index, eigenvectors,pc,_, weights,_ = indexes.principal_component_analysis(xtrain_fold,required_variance=0.90,
            normalizzation=True)
        
        ACI = xtest_fold @ eigenvectors[:,:pc]
        ACI1_test[idxTest] = ACI[:,0]
        ACI2_test[idxTest] = ACI[:,1]
        GACI_test[idxTest] = ACI @ weights

        pc_k[i] = pc

    pc_mean = np.mean(pc_k)
    pc_std = np.std(pc_k, ddof=1)

    ACI_p_1_test= normalize_index(ACI1_test)
    ACI_p_2_test= normalize_index(ACI2_test)
    GACI_p_test= normalize_index(GACI_test)

    corr_ACI1 = np.corrcoef(ACI1, ACI_p_1_test)[0, 1]
    corr_ACI2 = np.corrcoef(ACI2, ACI_p_2_test)[0, 1]
    corr_GACI = np.corrcoef(GACI, GACI_p_test)[0, 1]

    return pc_mean, pc_std, corr_ACI1, corr_ACI2, corr_GACI






  