import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from learn_truncated_PSGD import LearnDistribution, MyValueError

plt.style.use(['dark_background'])


def truncated_binomial_point_mass_estimator(binom_gen, point, epsilon=0.1, delta=0.01, m=None):

  if m is None:
    m = int( np.log(1/delta) / epsilon**2 )

  samples = np.array([binom_gen.rvs(1) for _ in range(m)])

  point_est = len( samples[samples == point] ) / m

  return point_est

def choose_points(samples_gen, samples, truncation_set):

    estimations = []
    set_len = len(truncation_set)

    for point in truncation_set:
        est = truncated_binomial_point_mass_estimator(samples_gen, point, m=int(samples/set_len)+1)
        estimations.append(est)

    max_indices = np.argsort(estimations)
    x = truncation_set[max_indices[-1]]
    y = truncation_set[max_indices[-2]]
    z = truncation_set[max_indices[-3]]

    return (x, y, z)

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


def learn_p_PSGD(samples_gen, truncation_set, alpha, \
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
        h = 1 / alpha**2
    print('\nStep size', round(h, 5))

    #samples
    M = 100 * int(1 / (alpha * epsilon**2) )
    total_samples = 0
    print('\nSamples:', M)

    #n estimation
    l_limit = 400
    u_limit = 1200
    print('Upper bound:', u_limit, 'Lower bound:', l_limit)
    n_values = range(l_limit, u_limit)

    x, y, z = choose_points(samples_gen, int(M/7), truncation_set)
    d = min( min(abs(x-y), abs(x-z)), abs(y-z))

    print('Estimation points', x, y, z)

    px_est = truncated_binomial_point_mass_estimator(samples_gen, x, m=int(2*M/7))
    py_est = truncated_binomial_point_mass_estimator(samples_gen, y, m=int(2*M/7))
    pz_est = truncated_binomial_point_mass_estimator(samples_gen, z, m=int(2*M/7))
    total_samples += M
    M = int(1 / (alpha * epsilon**2) )

    print('Estimated mass on points', px_est, py_est, pz_est)

    for n_est in n_values:

        par_est = empirical_estimator(m, samples_gen) / n_est
        nat_par_est = np.log(par_est / (1-par_est))
        print("Empirical Estimation:", round(par_est, 2), \
              "\nNatural Parameter Empirical Estimation:", round(nat_par_est, 5) )


        pSGD = LearnDistribution(binom, True, truncation_set, \
                                project, \
                                reverse_transformation, \
                                sufficient_statistics, \
                                other_params = [n_est])

        while True:
            try:
                par = pSGD.PSGD(M, h, samples_gen, (nat_par_est, B))
                total_samples += M
            except MyValueError:
                print('Out of bounds! Restarting...')
                continue
            break

        p_est = reverse_transformation(par)

        px = binom.pmf(x, n_est, p_est)
        py = binom.pmf(y, n_est, p_est)
        pz = binom.pmf(z, n_est, p_est)

        err1 = abs(pz_est/py_est - pz/py)
        err2 = abs(px_est/py_est - px/py)
        tolerance = epsilon / alpha
        print(tolerance)

        if printing:
          print('\nn:', n_est)
          print('p_est:', p_est)
          print(px, py, pz)
          print(err1, err2)

        if err1 < tolerance and err2 < tolerance:
          return ((n_est, p_est), total_samples)

    return ((1, 0), total_samples)
