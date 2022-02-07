import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from learning_truncated_binomial_algos.truncated_PSGD \
import LearnDistribution, MyValueError

exec(open("./styling.py").read())


def project(par, ball):

    c, B = ball

    proj_par = np.array([0.0, 0.0])

    #sigma
    if par[1] / c[1] < min(1 / B, B):
        par[1] = c[1] * min(1 / B, B)

    if abs(par[1]/c[1] - 1) <= B:
        proj_par[1] = par[1]
    else:
        proj_par[1] = par[1]/c[1] - 1
        proj_par[1] = (B / abs(proj_par[1])) * proj_par[1]
        proj_par[1] = c[1] * proj_par[1] + c[1]

    #mu
    if abs(par[0] - c[0]) <= B:
        proj_par[0] = par[0]
    else:
        proj_par[0] = par[0] - c[0]
        proj_par[0] = (B / abs(proj_par[0])) * proj_par[0]
        proj_par[0] += c[0]

    return proj_par

def sufficient_statistics(empirical_est, sample):

    #affine transform
    x = (sample - empirical_est[0]) / empirical_est[1]

    return np.array([x, -x**2 / 2])

def reverse_transformation(empirical_est, nat_par):

    mu = nat_par[0] * empirical_est[1]**2
    sigma_square = 1 / nat_par[1]
    return np.array([mu, np.sqrt(sigma_square)])


def empirical_estimator(m, gen):

    mu_est = 0
    sigma_est = 0
    for _  in range(m):
        mu_est += gen.rvs(1)

    for _ in range(m):
        sigma_est += (gen.rvs(1) - (mu_est/m) )**2

    return np.array([mu_est / m, np.sqrt(sigma_est / (m-1))])


def learn_truncated_binomial_PSGD(samples_gen, truncation_set, alpha, \
                                    B=None, h=None, M=None \
                                    epsilon=0.1, delta=0.01, printing=None):

    #parameter empirical estimators
    m = round( np.log(1 / delta)**2 * np.log(1 / alpha) / epsilon**2 ) + 2
    print('Samples for empirical estimation:', m)

    par_est = empirical_estimator(m, samples_gen)
    nat_par_est = np.array([ par_est[0] / par_est[1]**2, 1 / par_est[1]**2 ])
    print("Empirical Estimation:", round(par_est[0], 2), round((par_est[1]), 2), \
          "\nNatural Parameter Empirical Estimation:", round(nat_par_est[0], 5), \
          round(nat_par_est[1], 5) )

    #projection ball radius
    if B is None:
        B = (np.log(1 / alpha) / alpha**3) + 8
    print('\nBall radius:', round(B, 5))

    #hyperparameter, step size
    if h is None:
        h = 1 / alpha
    print('\nStep size', round(h, 5))

    #samples
    if M is None:
        M = 50 * int(1 / (alpha * epsilon**2) )
    print('\nSamples:', M)

    pSGD = LearnDistribution(norm, False, truncation_set, \
                                project, \
                                lambda par : reverse_transformation(par_est, par), \
                                lambda par : sufficient_statistics(par_est, par))

    while True:
        try:
            par = pSGD.PSGD(M, h, samples_gen, (nat_par_est, B), printing)
        except MyValueError:
            print('Out of bounds! Restarting...')
            continue
        break

    #printing
    true_par = reverse_transformation(par_est, par)
    p = 1 - ( true_par[1]**2 / true_par[0])
    true_par = np.array([true_par[0] / p, p])
    print('\nTrue Parameters Estimation:', int(true_par[0]), round(true_par[1], 5))

    return (true_par, M)
