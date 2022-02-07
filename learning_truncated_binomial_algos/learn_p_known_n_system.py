from scipy.special import binom as binomial_coeff
import numpy as np
from scipy.stats import binom

def truncated_binomial_point_mass_estimator(binom_gen, point, epsilon=0.1, delta=0.01, m=None):

  if m is None:
    m = int( np.log(1/delta) / epsilon**2 )

  samples = np.array([binom_gen.rvs(1) for _ in range(m)])

  point_est = len( samples[samples == point] ) / m

  return point_est

def choose_points(samples_gen, samples, truncation_set):

    estimations = []
    set_len = len(truncation_set)

    samples_set = [samples_gen.rvs(1) for _ in range(samples)]

    for point in truncation_set:
        est = samples_set.count(point)
        estimations.append(est)

    max_indices = np.argsort(estimations)
    x = truncation_set[max_indices[-1]]
    y = truncation_set[max_indices[-2]]
    z = truncation_set[max_indices[-3]]

    return (x, y, z)


def learn_truncated_binomial_known_n_system_solution(n, samples_gen, truncation_set, alpha, \
                                                     B=None, h=None,\
                                                     epsilon=0.1, delta=0.01, printing=False):

  #samples
  M = n * int(1 / (alpha * epsilon**2) )
  if printing:
      print('Samples', M)

  #choose point for estimtion
  x, y, z = choose_points(samples_gen, int(M/7), truncation_set)
  if printing:
      print('Estimation points', x, y, z)

  #estimate mass on points
  px_est = truncated_binomial_point_mass_estimator(samples_gen, x, m=int(2*M/7))
  py_est = truncated_binomial_point_mass_estimator(samples_gen, y, m=int(2*M/7))
  pz_est = truncated_binomial_point_mass_estimator(samples_gen, z, m=int(2*M/7))
  if printing:
      print('Estimated mass on points', px_est, py_est, pz_est)

  #estimate p
  nom = px_est * binomial_coeff(n, y) #binomial coefficient
  denom = py_est * binomial_coeff(n, x)
  f = ( nom / denom )**(1/(x-y))
  p_est = f / ( 1 + f )
  if printing:
      print('Estimated parameter', p_est)

  return (p_est, M)
