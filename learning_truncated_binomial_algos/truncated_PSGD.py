import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete, norm

plt.style.use(['dark_background'])


class MyValueError(ValueError): pass

class LearnDistribution():

    def __init__(self, distr, pmf, set, \
                       project = lambda x, y : x, \
                       reverse_tr = lambda x : x, \
                       suff_stats = lambda x : x, \
                       other_params = None):
        self.distribution = distr
        self.pmf = pmf
        self.set = set
        self.project = project
        self.reverse_tr = reverse_tr
        self.suff_stats = suff_stats
        self.other_params = other_params

    def truncated_distribution(self, params):

        if self.other_params is not None:
            if isinstance(params, list):
                params_list = [*self.other_params, *params]
            else:
                params_list = [*self.other_params, params]
        else:
            params_list = params

        if self.pmf:
            probs = self.distribution(*params_list).pmf(self.set)
        else:
            probs = self.distribution(*params_list).pdf(self.set)
        probs = probs / sum(probs)

        return rv_discrete(name='trun_distr', values=(self.set, probs))

    def reverse_transform(self, par):

        return self.reverse_tr(par)

    def sufficient_statistics(self, arg):

        return self.suff_stats(arg)

    def project(self, a, b):

        return self.project(a, b)

    def PSGD(self, samples_size, step_size, samples_oracle, ball, printing=None):

        (nat_par_est, ball_radius) = ball

        #initialization
        nat_par = nat_par_est
        if isinstance(nat_par_est, list):
            nat_par_mean = np.zeros(len(nat_par_est))
        else:
            nat_par_mean = 0

        for i in range(1, samples_size+1):

            #true distribution mean value estimator
            x = samples_oracle.rvs(1)

            #estimated distribution
            par = self.reverse_transform(nat_par)
            try:
                est_distr = self.truncated_distribution(par)
            except:
                raise MyValueError

            #estimated distribution mean value estimator
            y = est_distr.rvs(1)

            #gradient estimator
            v = - self.sufficient_statistics(x) + self.sufficient_statistics(y)
            #new estimator
            nat_par = nat_par - v / (i*step_size)
            #project to feasible set
            nat_par = self.project(nat_par, ball)

            #print progess
            if printing is not None and i % int(samples_size / 10) == 0:
                print("Iteration:", i, "Estimation:", nat_par,\
                      "Mean of estimations", nat_par_mean / i)

            nat_par_mean += nat_par

        return nat_par_mean / samples_size
