import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from data_utilities.utilities import \
    ecdf, \
    pearson_r, \
    set_seaborn_styling, \
    perform_bernoulli_trials, \
    successive_poisson


def generate_randoms_and_plot():
    # Seed the random number generator
    np.random.seed(42)

    # Initialize random numbers: random_numbers
    random_numbers = np.empty(100000)

    # Generate random numbers by looping over range(100000)
    for i in range(100000):
        random_numbers[i] = np.random.random()

    # Plot a histogram
    _ = plt.hist(random_numbers)

    # Show the plot
    plt.show()


def perform_bernoulli_trials_example():
    trials = np.sum(perform_bernoulli_trials(100, .5))
    print(f"Bernoulli Trials Sum: {trials}")
    return trials


def how_many_defaults_to_expect():
    # Seed random number generator
    np.random.seed(42)

    # Initialize the number of defaults: n_defaults
    n_defaults = np.empty(1000)

    # Compute the number of defaults
    for i in range(1000):
        n_defaults[i] = perform_bernoulli_trials(100, 0.05)

    # Plot the histogram with default number of bins; label your axes
    _ = plt.hist(n_defaults, density=True)
    _ = plt.xlabel('number of defaults out of 100 loans')
    _ = plt.ylabel('probability')

    # Show the plot
    plt.show()


def will_the_banks_fail():
    # Seed random number generator
    np.random.seed(42)

    # Initialize the number of defaults: n_defaults
    n_defaults = np.empty(1000)

    # Compute the number of defaults
    for i in range(1000):
        n_defaults[i] = perform_bernoulli_trials(100, 0.05)

    # Compute ECDF: x, y
    x, y = ecdf(n_defaults)

    # Plot the ECDF with labeled axes
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('Number of Defaults')
    _ = plt.ylabel('Probability')

    # Show the plot
    plt.show()

    # Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
    n_lose_money = np.sum(n_defaults >= 10)

    # Compute and print probability of losing money
    print('Probability of losing money =', n_lose_money / len(n_defaults))


def sample_from_binomial_distribution():
    # Random number of trues out of four with 50% chance
    np.random.seed(42)
    print(f"Random number of trues out of four with 50% chance: {np.random.binomial(4, 0.5)}")
    print(f"Set of ten trials of the above: {np.random.binomial(4, 0.5, size=10)}\n")


def plot_binomial_pmf():
    samples = np.random.binomial(60, 0.1, size=10000)
    n = 60
    p = 0.1
    x, y = ecdf(samples)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('number of successes')
    _ = plt.ylabel('CDF')
    plt.margins(0.02)
    plt.show()


def banks_sample_from_binomial_distribution():
    # Take 10,000 samples out of the binomial distribution: n_defaults
    n_defaults = np.random.binomial(n=100, p=0.05, size=10000)

    # Compute CDF: x, y
    x, y = ecdf(n_defaults)

    # Plot the CDF with axis labels
    _ = plt.plot(x, y, marker='.', linestyle='none')
    plt.margins(0.02)
    plt.xlabel('Number of Defaults')
    plt.ylabel('CDF')

    # Show the plot
    plt.show()


def banks_plot_binomial_pmf():
    # Take 10,000 samples out of the binomial distribution: n_defaults
    n_defaults = np.random.binomial(n=100, p=0.05, size=10000)

    # Compute bin edges: bins
    bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

    # Generate histogram
    _ = plt.hist(n_defaults, density=True, bins=bins)

    # Label axes
    _ = plt.xlabel('Number of Defaults')
    _ = plt.ylabel('CDF')

    # Show the plot
    plt.show()


def sample_poisson_cdf():
    samples = np.random.poisson(6, size=10000)
    x, y = ecdf(samples)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('number of successes')
    _ = plt.ylabel('CDF')
    plt.margins(0.02)
    plt.show()


def show_relationship_between_binomial_and_poisson():
    # Draw 10,000 samples out of Poisson distribution: samples_poisson
    samples_poisson = np.random.poisson(10, 10000)

    # Print the mean and standard deviation
    print('Poisson:     ', np.mean(samples_poisson), np.std(samples_poisson))

    # Specify values of n and p to consider for Binomial: n, p
    n = [20, 100, 1000]
    p = [0.5, 0.1, 0.01]

    # Draw 10,000 samples for each n,p pair: samples_binomial
    for i in range(3):
        samples_binomial = np.random.binomial(n[i], p[i], size=10000)

        # Print results
        print('n =', n[i], 'Binom:', np.mean(samples_binomial), np.std(samples_binomial))


def was_2015_anomalous():
    # Draw 10,000 samples out of Poisson distribution: n_nohitters
    n_nohitters = np.random.poisson(251 / 115, size=10000)

    # Compute number of samples that are seven or greater: n_large
    n_large = np.sum(n_nohitters >= 7)

    # Compute probability of getting seven or more: p_large
    p_large = n_large / 10000

    # Print the result
    print('\nProbability of seven or more no-hitters:', p_large)


def distribution_of_nohitters_and_cycles():
    # Draw samples of waiting times: waiting_times
    waiting_times = successive_poisson(764, 715, size=100000)

    # Make the histogram
    _ = plt.hist(waiting_times, bins=100, density=True, histtype='step')

    # Label axes
    _ = plt.xlabel('time (games)')
    _ = plt.ylabel('PDF')

    # Show the plot
    plt.show()


def main():
    set_seaborn_styling()
    generate_randoms_and_plot()
    perform_bernoulli_trials_example()
    how_many_defaults_to_expect()
    will_the_banks_fail()
    sample_from_binomial_distribution()
    plot_binomial_pmf()
    banks_sample_from_binomial_distribution()
    banks_plot_binomial_pmf()
    sample_poisson_cdf()
    show_relationship_between_binomial_and_poisson()
    was_2015_anomalous()
    distribution_of_nohitters_and_cycles()


if __name__ == '__main__':
    main()
