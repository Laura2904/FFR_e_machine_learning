import numpy as np

def assessment_metrics(data, binary_class, positive_class=1, negative_class=0, alpha=None):
    """
    Compute Accuracy, Sensitivity e Specificity at different thresholds

    Parameters
    ----------
    data : array-like
        Predicted probability (N,).
    binary_class : array-like
        Real class (N,).
    positive_class : int, optional
        label of the positive class (default 1).
    negative_class : int, optional
        label of the negative class(default 0).
    alpha : array-like, optional
        threshold values (default np.linspace(1, 0, 101)).

    Returns
    -------
    accuracy : ndarray
    sensitivity : ndarray
    specificity : ndarray
    """
    if alpha is None:
        alpha = np.linspace(1, 0, 101)

    data = np.asarray(data).ravel()
    binary_class = np.asarray(binary_class).ravel()
    alpha = np.asarray(alpha)[:, None]  

    #Boolean value for the real class
    is_pos = (binary_class == positive_class)
    is_neg = (binary_class == negative_class)

    total_pos = np.sum(is_pos)
    total_neg = np.sum(is_neg)
    total_samples = len(binary_class)

    # Matrix of prediction for all threshold
    preds_pos = (data >= alpha)  # True if the class is positive

    # confusion matrix
    TP = np.sum(preds_pos & is_pos, axis=1)
    TN = np.sum(~preds_pos & is_neg, axis=1)

    # compute metrics
    accuracy = (TP + TN) / total_samples if total_samples > 0 else np.zeros(len(alpha))
    sensitivity = TP / total_pos if total_pos > 0 else np.zeros(len(alpha))
    specificity = TN / total_neg if total_neg > 0 else np.zeros(len(alpha))

    return accuracy, sensitivity, specificity