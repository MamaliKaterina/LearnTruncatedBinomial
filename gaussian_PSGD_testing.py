import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete, binom
from learning_truncated_binomial_algos.truncated_distribution \
import learn_binomial_PSGD

exec(open("./styling.py").read())

#scaling S
#setting
n = 1000
p = 0.3
domain = np.array(range(n+1))
epsilon = 0.08

def truncated_binomial(gen, set):

    probs = gen.pmf(S)
    probs = probs / sum(probs)
    return rv_discrete(name='trun_binomial', values=(set, probs))

Sets = np.array([
                 [list(range(285, 315)),\
                  list(range(10, 308)),\
                  list(range(10,295))+list(range(306,500))],
                 [list(range(290, 310)),\
                  list(range(10, 301)),\
                  list(range(10,291))+list(range(310, 500)) ],
                 [list(range(293, 308)),\
                  list(range(10, 297)),\
                  list(range(10,288))+list(range(312, 500)) ]
                 ])
accuracy = np.zeros((3,3))
n_ests = np.ones((3,3))
p_ests = np.ones((3,3))


for i in range(3):
  for j in range(3):

      S = Sets[i][j]
      true_dist = truncated_binomial(binom(n, p), S)
      a = sum(binom.pmf(S, n, p))
      print('\n\nTrue parameters:', n, round(p, 5),\
            '\nTrue gaussian parameters:', round(n*p, 2), round(n*p*(1-p), 2),\
            '\nTrue Natural parameters:', round(n*p / (n*p*(1-p)), 1), round(1 / (n*p*(1-p)), 5),\
            '\nAlpha:', round(a, 2)
            )
      print()

      #estimation
      (n_est, p_est), samples =\
        learn_binomial_PSGD(true_dist, S, a,\
                                      epsilon=epsilon, printing=2)

      #TV distance
      true_pmf = binom.pmf(domain, n, p)
      est_pmf = binom.pmf(domain, n_est, p_est)
      TV_distance = round( sum( abs(true_pmf - est_pmf) ) / 2, 4)
      print('TV distance:', TV_distance)

      #plotting
      plt.title('TV distance: '+str(round(TV_distance, 3)))
      plt.plot(domain[200:400], true_pmf[200:400], label='True')
      plt.plot(domain[200:400], est_pmf[200:400], label='Estimated')
      plt.legend()
      plt.show()

      accuracy[i][j] = TV_distance
      n_ests[i][j] = n_est
      p_ests[i][j] = p_est


fig = plt.figure(figsize=(20,12.8))
axs = fig.subplots(3, 3, sharex=True, sharey=True)
plt.subplots_adjust(left=0.05, bottom=0.03, right=0.98, top=0.92, wspace=0.1, hspace=0.15)
fig.suptitle('Bin('+str(n)+', '+str(round(p,2))+')   ??='+str(epsilon),\
              y=0.98, size='xx-large', weight='black')

for i in range(3):
  for j in range(3):

      S = Sets[i][j]
      a = sum( binom.pmf(S, n, p) )

      #plotting grid
      axs[i, j].plot(S, binom.pmf(S, n, p),\
                     linestyle='None', color='#c54646', marker='o', markersize=4)
      axs[i, j].plot(domain, binom.pmf(domain, n_ests[i][j], p_ests[i][j]),\
                     linestyle='None', marker='o', color='#f28165', markersize=4)
      axs[i, j].set_title( 'TV distance='+str(accuracy[i][j]), size='xx-large' )

      if j == 0:
          axs[i, j].set_ylabel('??='+str(round(a, 2)), size='x-large')

plt.xlim([230,370])
plt.ylim([-0.0001, 0.03])

labels = ['True Distr', 'Estimated Distr']
plt.figlegend(labels, fontsize='large')

plt.savefig('images/gaussian_PSGD_testing.pdf')
plt.show()
