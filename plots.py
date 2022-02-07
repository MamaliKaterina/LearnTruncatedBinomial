import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

from scipy.stats import norm, gamma

domain = np.linspace(0, 50, 1000)

#gaussian
mu = 10
sigma = np.sqrt(10)

gaussian = norm.pdf(domain, mu, sigma)

#gamma
alpha = 10
beta = 1

gamma = gamma.pdf(domain, alpha, 0, 1/beta)

plt.plot(domain, gaussian, 'r', label='N('+str(mu)+','+str(round(sigma**2, 2))+')')
plt.plot(domain, gamma, 'b', label='Gamma('+str(alpha)+','+str(beta)+')')

plt.title('Gaussian VS Gamma')
plt.legend()
plt.savefig('GaussianVSGamma')
plt.show()
