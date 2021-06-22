import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from data_utilities.utilities import \
    ecdf, \
    pearson_r, \
    set_seaborn_styling, \
    perform_bernoulli_trials


def are_the_stakes_normally_distributed(belmont_no_outliers):
    # Compute mean and standard deviation: mu, sigma
    mu = np.mean(belmont_no_outliers)
    sigma = np.std(belmont_no_outliers)

    # Sample out of a normal distribution with this mu and sigma: samples
    samples = np.random.normal(mu, sigma, size=10000)

    # Get the CDF of the samples and of the data
    x_theor, y_theor = ecdf(samples)
    x, y = ecdf(belmont_no_outliers)

    # Plot the CDFs and show the plot
    _ = plt.plot(x_theor, y_theor)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('Belmont winning time (sec.)')
    _ = plt.ylabel('CDF')
    plt.show()


def what_are_the_chances_of_another_secretariat(belmont_no_outliers):
    # Compute mean and standard deviation: mu, sigma
    mu = np.mean(belmont_no_outliers)
    sigma = np.std(belmont_no_outliers)

    # Take a million samples out of the Normal distribution: samples
    samples = np.random.normal(mu, sigma, size=1000000)

    # Compute the fraction that are faster than 144 seconds: prob
    prob = np.sum(samples <= 144) / 1000000

    # Print the result
    print('Probability of besting Secretariat:', prob)


def main():
    belmont_no_outliers = np.array([148.51, 146.65, 148.52, 150.7, 150.42, 150.88, 151.57, 147.54,
                                    149.65, 148.74, 147.86, 148.75, 147.5, 148.26, 149.71, 146.56,
                                    151.19, 147.88, 149.16, 148.82, 148.96, 152.02, 146.82, 149.97,
                                    146.13, 148.1, 147.2, 146.0, 146.4, 148.2, 149.8, 147.0,
                                    147.2, 147.8, 148.2, 149.0, 149.8, 148.6, 146.8, 149.6,
                                    149.0, 148.2, 149.2, 148.0, 150.4, 148.8, 147.2, 148.8,
                                    149.6, 148.4, 148.4, 150.2, 148.8, 149.2, 149.2, 148.4,
                                    150.2, 146.6, 149.8, 149.0, 150.8, 148.6, 150.2, 149.0,
                                    148.6, 150.2, 148.2, 149.4, 150.8, 150.2, 152.2, 148.2,
                                    149.2, 151.0, 149.6, 149.6, 149.4, 148.6, 150.0, 150.6,
                                    149.2, 152.6, 152.8, 149.6, 151.6, 152.8, 153.2, 152.4,
                                    152.2])
    are_the_stakes_normally_distributed(belmont_no_outliers)
    what_are_the_chances_of_another_secretariat(belmont_no_outliers)


if __name__ == '__main__':
    main()
