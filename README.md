# Experimental Algorithms for Learning Binomial Distribution from Truncated Samples


## About
This library contains the implementation od three different algorithms that can be used to leaarn a Binomial distribution from Truncated Samples.
They were implemented as part of my diploma thesis and the proofs for their correctness can be found here.

Specifically you can find the following three methods:

* `learn_binomial_PSGD`: retrieves both parameters of the Binomial `n`,`p`.

* `learn_p_known_n_PSGD`: for known parameter `n` returns estimation for `p`.

* `learn_p_known_n_system`: for known parameter `n` returns estimation for `p` (slower).

Three functions that use each of the above methods are also provided as examples.


## Author
Katerina Mamali


## Dependencies
* NumPy
* Matplotlib.pyplot
* SciPy.stats

## Installation
Run the following commands:

* `python3 setup.py bdist_wheel`

* `pip install <location of .whl file>`

## Usage
Below we present the existing function along with their parameters and explain their meaning.
1. `learn_binomial_PSGD`
  * `samples_gen`:
  A generator of samples from the distribution in question.
  Must be samples from a Binomial distribution, orelse there is no guarantee for the output of the algorithm.
  It can be any distribution generator with a method `rvs` to return samples of this distribution.
  * `truncation set`:
  A list or array of a subset of the distribution domain.
  * `alpha`:
  The truncation set mass in the distribution in question.
  * `B`:
  The radius of the ball which is the set on which the algorithm projects the solutions.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `h`:
  The learning rate.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `M`:
  Number of samples used by the algorithm.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `epsilon`:
  Controls the accuracy achieved by the algorithm.
  Default `0.1`.
  * `delta`:
  Controls the succes probability of the algorithm.
  Default `99%`.
  * `printing`:
  Controls printing of intermediate results while the program searches for the distribution's parameters.
  If enabled it will print 10 updates for the state of search.
  Default `None`.

2. `learn_p_known_n_PSGD`
  * `n`:
  The parameter of the Binomial.
  * `samples_gen`:
  A generator of samples from the distribution in question.
  Must be samples from a Binomial distribution, orelse there is no guarantee for the output of the algorithm.
  It can be any distribution generator with a method `rvs` to return samples of this distribution.
  * `truncation set`:
  A list or array of a subset of the distribution domain.
  * `alpha`:
  The truncation set mass in the distribution in question.
  * `B`:
  The radius of the ball which is the set on which the algorithm projects the solutions.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `h`:
  The learning rate.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `M`:
  Number of samples used by the algorithm.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `epsilon`:
  Controls the accuracy achieved by the algorithm.
  Default `0.1`.
  * `delta`:
  Controls the succes probability of the algorithm.
  Default `99%`.
  * `printing`:
  Controls printing of intermediate results while the program searches for the distribution's parameters.
  If enabled it will print 10 updates for the state of search.
  Default `None`.

3. `learn_p_known_n_system`
  * `n`:
  The parameter of the Binomial.
  * `samples_gen`:
  A generator of samples from the distribution in question.
  Must be samples from a Binomial distribution, orelse there is no guarantee for the output of the algorithm.
  It can be any distribution generator with a method `rvs` to return samples of this distribution.
  * `truncation set`:
  A list or array of a subset of the distribution domain.
  * `alpha`:
  The truncation set mass in the distribution in question.
  * `M`:
  Number of samples used by the algorithm.
  If not specified by the user the algorithm uses a function of `epsilon`, `alpha`, `delta` to calculate this value.
  Default `None`.
  * `epsilon`:
  Controls the accuracy achieved by the algorithm.
  Default `0.1`.
  * `delta`:
  Controls the succes probability of the algorithm.
  Default `99%`.
  * `printing`:
  Controls printing of intermediate results while the program searches for the distribution's parameters.
  If enabled it will print 10 updates for the state of search.
  Default `None`.

## Reference
