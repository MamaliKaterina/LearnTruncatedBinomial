# Experimental Algorithms for Learning Binomial Distribution from Truncated Samples
***

## About
---
This library contains the implementation od three different algorithms that can be used to leaarn a Binomial distribution from Truncated Samples.
They were implemented as part of my diploma thesis and the proofs for their correctness can be found here.

Specifically you can find the following three methods:

* `learn_binomial_PSGD`: retrieves both parameters of the Binomial $n, p$.

* `learn_p_known_n_PSGD`: for known parameter $n$ returns estimation for $p$.

* `learn_p_known_n_system`: for known parameter $n$ returns estimation for $p$ (slower).

Three functions that use each of the above methods are also provided as examples.


## Author
---
Katerina Mamali


## Dependencies
---
* NumPy
* Matplotlib.pyplot
* SciPy.stats

## Installation
---
Run the following commands:

* `python3 setup.py bdist_wheel`

* `pip install <location of .whl file>`

## Usage
---
1.

2.

3.

## Reference
