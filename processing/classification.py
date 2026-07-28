# Modelli di classificazione binaria per identificare la presenza/assenza delle FFR

import numpy as np

def Newton_Raphson(data,binary_class,lam,maxiter, toll):

    #the function implements the Newton Raphson method  with the maximum likelihood criterion
    # in order to find the best coefficients for the binary logistic regression model.

    column = data.shape[1]
    beta_old = np.zeros(column)
    iter = 0
    error = np.inf
    log_lik_old = - np.inf
    binary_class = binary_class.ravel()

    while error > toll and iter < maxiter:

        iter += 1

        #obtain the beta coefficients
        element1 = np.exp(-data @ beta_old)
        element2 = 1 / (1+element1)
        Win = element2*(1-element2)
        Lambda = lam*np.eye(column)
        Lambda[0,0] = 0

        penalty = lam* beta_old
        penalty[0] = 0

        H = data.T @ ( data * Win ) + Lambda
        g = data.T @ ( binary_class - element2) - penalty

        delta = np.linalg.solve(H, g)
        beta_new = beta_old + delta
        beta_old = beta_new

        #calculate the log likehood
        element3 = 1 / ( 1 + np.exp(-data @ beta_new))
        eps = 1e-12
        element4 = np.clip(element3, eps, 1-eps)
        log_likelihood = (
            np.sum(
                binary_class * np.log(element4)
                + (1 - binary_class) * np.log(1 - element4)
            )
            - (lam / 2) * np.sum(beta_new[1:] ** 2)
        )

        error = np.abs(log_likelihood - log_lik_old)
        log_lik_old = log_likelihood

    beta = beta_new
    return beta



