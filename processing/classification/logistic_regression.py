import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from processing.assessment import assessment_metrics as am

def sigmoid(z):
    """computation of sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def Newton_Raphson(data, binary_class, lam, maxiter=500, toll=1e-3):
    """
    Newton-Raphson for Logistic Regression with L2 (Ridge) regularization.
    """
    column = data.shape[1]
    beta_old = np.zeros(column)
    binary_class = binary_class.ravel()
    
    log_lik_old = -np.inf

    for _ in range(maxiter):
        # Predicted probability = P(y=1|x)
        p = sigmoid(data @ beta_old)
        
        # weights W = p * (1 - p)
        Win = p * (1 - p)
        
        # Regularization matrix 
        Lambda = lam * np.eye(column)
        Lambda[0, 0] = 0

        penalty = lam * beta_old
        penalty[0] = 0

        # Hessiana: H = X^T * W * X + Lambda
        H = data.T @ (data * Win[:, None]) + Lambda
        
        # Gradient: g = X^T * (y - p) - Lambda * beta
        g = data.T @ (binary_class - p) - penalty

        # Update Newton-Raphson: delta = H^-1 * g
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback in case of singular matrix
            delta = np.linalg.pinv(H) @ g

        beta_new = beta_old + delta

        #Log-Likelihood 
        p_new = sigmoid(data @ beta_new)
        eps = 1e-12
        p_clipped = np.clip(p_new, eps, 1 - eps)
        
        log_likelihood = (
            np.sum(binary_class * np.log(p_clipped) + (1 - binary_class) * np.log(1 - p_clipped))
            - (lam / 2) * np.sum(beta_new[1:] ** 2)
        )

        if np.abs(log_likelihood - log_lik_old) < toll:
            beta_old = beta_new
            break

        beta_old = beta_new
        log_lik_old = log_likelihood

    return beta_old


def binomial_logistic_regression(training_data, test_data, k_fold = 5):
    """
    It computes the Logistic Regression with K-Fold CV for lambda selection.
    """
    # Divide feature and class
    training_class = training_data[:, -1]
    test_class = test_data[:, -1]

    training_feature = training_data[:, :-1]
    test_feature = test_data[:, :-1]

    #Z-score
    mean = np.mean(training_feature, axis=0)
    std = np.std(training_feature, axis=0, ddof=1)
    std[std == 0] = 1  # avoid division by zero

    training_feature = (training_feature - mean) / std
    test_feature = (test_feature - mean) / std

    # Add intercept
    xtrain = np.hstack((np.ones((training_feature.shape[0], 1)), training_feature))
    xtest = np.hstack((np.ones((test_feature.shape[0], 1)), test_feature))

    toll = 1e-3
    maxiter = 500
    lambdas = [0.0001, 0.001, 0.01, 0.1]
    l = len(lambdas)

    mean_accuracy = np.zeros(l)
    mean_loss = np.zeros(l)

    cv = StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=42)

    # Cross-Validation to find the best lambda
    for h, lam in enumerate(lambdas):
        loss_beta = np.zeros(k_fold)
        accuracy_beta = np.zeros(k_fold)

        for i, (idxTrain, idxTest) in enumerate(cv.split(xtrain)):
            xtrain_fold, ytrain_fold = xtrain[idxTrain], training_class[idxTrain]
            xtest_fold, ytest_fold = xtrain[idxTest], training_class[idxTest]

            beta_fold = Newton_Raphson(xtrain_fold, ytrain_fold, lam, maxiter, toll)

            # Prediction in the test fold
            ptest_fold = sigmoid(xtest_fold @ beta_fold)
            pred_class = (ptest_fold >= 0.5).astype(int)

            # Log-Loss / Cross-Entropy
            eps = 1e-12
            ptest_fold_clipped = np.clip(ptest_fold, eps, 1 - eps)
            loss = -np.mean(
                ytest_fold.ravel() * np.log(ptest_fold_clipped)
                + (1 - ytest_fold.ravel()) * np.log(1 - ptest_fold_clipped)
            )

            loss_beta[i] = loss
            accuracy_beta[i] = np.mean(pred_class == ytest_fold)

        mean_accuracy[h] = np.mean(accuracy_beta)
        mean_loss[h] = np.mean(loss_beta)

    # Selection of the best lambda which minimizes the loss
    ind = np.argmin(mean_loss)
    lambda_opt = lambdas[ind]

    # final training the whole training set
    bestbeta = Newton_Raphson(xtrain, training_class, lambda_opt, maxiter, toll)

    # final prediction
    ptrain = sigmoid(xtrain @ bestbeta)
    ptest = sigmoid(xtest @ bestbeta)

    # assessment
    positiveclass = 1
    negativeclass = 0
    alpha = np.linspace(1, 0, 101)

    train_accuracy, train_sensibility, train_specificity = am(ptrain, training_class, positiveclass, negativeclass, alpha)
    test_accuracy, test_sensibility, test_specificity = am(ptest, test_class, positiveclass, negativeclass, alpha)

    model_accuracy = [train_accuracy, test_accuracy]
    model_sensibility = [train_sensibility, test_sensibility]
    model_specificity = [train_specificity, test_specificity]

    return bestbeta, model_accuracy, model_sensibility, model_specificity

