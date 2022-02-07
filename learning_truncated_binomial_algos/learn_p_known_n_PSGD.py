import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from learning_truncated_binomial_algos.learn_truncated_PSGD \
import LearnDistribution, MyValueError

plt.style.use(['dark_background'])


def project(par, ball):

    c, B = ball

    proj_par = 0.0

    #mu
    if abs(par - c) <= B:
        proj_par = par
    else:
        proj_par = par - c
        proj_par = (B / abs(proj_par)) * proj_par
        proj_par += c

    return proj_par

def sufficient_statistics(sample):

    return sample

def reverse_transformation(nat_par):

    return 1 / (1 + np.e**(-nat_par))


def empirical_estimator(m, gen):

    p_est = 0
    for _  in range(m):
        p_est += gen.rvs(1)

    return p_est / m


def learn_truncated_binomial_known_n_PSGD(n, samples_gen, truncation_set, alpha, \
                                          B=None, h=None, \
                                          epsilon=0.1, delta=0.01, printing=None):

    #parameter empirical estimators
    m = round( np.log(1 / delta)**2 * np.log(1 / alpha) / epsilon**2 ) + 2
    print('Samples for empirical estimation:', m)

    #projection ball radius
    if B is None:
        B = (np.log(1 / alpha) / alpha)
    print('\nBall radius:', round(B, 5))

    #hyperparameter, step size
    if h is None:
        h = 1000
    print('\nStep size', round(h, 5))

    #samples
    M = 10 * int(1 / (alpha * epsilon**2) )
    print('\nSamples:', M)

    #empirical estimation
    par_est = empirical_estimator(m, samples_gen) / n
    nat_par_est = np.log(par_est / (1-par_est))
    print("Empirical Estimation:", round(par_est, 5), \
          "\nNatural Parameter Empirical Estimation:", round(nat_par_est, 5) )

    #pSGD
    pSGD = LearnDistribution(binom, True, truncation_set, \
                             project, \
                             reverse_transformation, \
                             sufficient_statistics, \
                             other_params = [n])

    while True:
        try:
            par = pSGD.PSGD(M, h, samples_gen, (nat_par_est, B))
        except MyValueError:
            print('Out of bounds! Restarting...')
            continue
        break

    p_est = reverse_transformation(par)
    print('Parameter Estimation:', p_est)

    return (p_est, M)
