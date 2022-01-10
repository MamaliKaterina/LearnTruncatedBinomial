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

    for point in truncation_set:
        est = truncated_binomial_point_mass_estimator(samples_gen, point, m=int(samples/set_len))
        estimations.append(est)

    max_indices = np.argsort(estimations)
    x = truncation_set[max_indices[-1]]
    y = truncation_set[max_indices[-2]]
    z = truncation_set[max_indices[-3]]

    return (x, y, z)


def learn_truncated_binomial_system_solution(N, samples_gen, truncation_set, alpha, \
                                             B=None, h=None,\
                                             epsilon=0.1, delta=0.01, printing=False):

  #initialize distribution
  l_limit = 400
  u_limit = 1200
  print('Upper bound:', u_limit, 'Lower bound:', l_limit)
  n_values = range(l_limit, u_limit)
  M = 1000 * int(1 / (alpha * epsilon**2) )
  print('Samples', M)


  x, y, z = choose_points(samples_gen, int(M/7), truncation_set)
  d = min( min(abs(x-y), abs(x-z)), abs(y-z))

  print('Estimation points', x, y, z)

  px_est = truncated_binomial_point_mass_estimator(samples_gen, x, m=int(2*M/7))
  py_est = truncated_binomial_point_mass_estimator(samples_gen, y, m=int(2*M/7))
  pz_est = truncated_binomial_point_mass_estimator(samples_gen, z, m=int(2*M/7))

  print('Estimated mass on points', px_est, py_est, pz_est)

  for n_est in n_values:

    nom = px_est * binomial_coeff(n_est, y) #binomial coefficient
    denom = py_est * binomial_coeff(n_est, x)
    f = ( nom / denom )**(1/(x-y))
    p_est = f / ( 1 + f )

    px = binom.pmf(x, n_est, p_est)
    py = binom.pmf(y, n_est, p_est)
    pz = binom.pmf(z, n_est, p_est)

    err1 = abs(pz_est/py_est - pz/py)
    err2 = abs(px_est/py_est - px/py)
    tolerance = epsilon
    print(tolerance)

    if printing:
      print('\nn:', n_est)
      print('p_est:', p_est, f, nom, denom)
      print(px, py, pz)
      print(err1, err2)

    if err1 < tolerance and err2 < tolerance:
      return ((n_est, p_est), M)


  return ((N, 0), M)
