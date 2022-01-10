from setuptools import find_packages, setup

setup(
    name='mypythonlib',
    packages=find_packages(include=['learning_truncated_binomial_algos']),
    version='0.1.0',
    description='Experimental Algorithms for Learning a Binomial Distribution\
                 using Truncated Samples',
    author='Katerina',
    license='MIT',
    install_requires=[],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)
