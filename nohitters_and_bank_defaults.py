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
    successive_poisson, \
    draw_bs_reps, diff_of_means, draw_perm_reps

nohitter_times = np.array([843, 1613, 1101, 215, 684, 814, 278, 324, 161, 219, 545,
                           715, 966, 624, 29, 450, 107, 20, 91, 1325, 124, 1468,
                           104, 1309, 429, 62, 1878, 1104, 123, 251, 93, 188, 983,
                           166, 96, 702, 23, 524, 26, 299, 59, 39, 12, 2,
                           308, 1114, 813, 887, 645, 2088, 42, 2090, 11, 886, 1665,
                           1084, 2900, 2432, 750, 4021, 1070, 1765, 1322, 26, 548, 1525,
                           77, 2181, 2752, 127, 2147, 211, 41, 1575, 151, 479, 697,
                           557, 2267, 542, 392, 73, 603, 233, 255, 528, 397, 1529,
                           1023, 1194, 462, 583, 37, 943, 996, 480, 1497, 717, 224,
                           219, 1531, 498, 44, 288, 267, 600, 52, 269, 1086, 386,
                           176, 2199, 216, 54, 675, 1243, 463, 650, 171, 327, 110,
                           774, 509, 8, 197, 136, 12, 1124, 64, 380, 811, 232,
                           192, 731, 715, 226, 605, 539, 1491, 323, 240, 179, 702,
                           156, 82, 1397, 354, 778, 603, 1001, 385, 986, 203, 149,
                           576, 445, 180, 1403, 252, 675, 1351, 2983, 1568, 45, 899,
                           3260, 1025, 31, 100, 2055, 4043, 79, 238, 3931, 2351, 595,
                           110, 215, 0, 563, 206, 660, 242, 577, 179, 157, 192,
                           192, 1848, 792, 1693, 55, 388, 225, 1134, 1172, 1555, 31,
                           1582, 1044, 378, 1687, 2915, 280, 765, 2819, 511, 1521, 745,
                           2491, 580, 2072, 6450, 578, 745, 1075, 1103, 1549, 1520, 138,
                           1202, 296, 277, 351, 391, 950, 459, 62, 1056, 1128, 139,
                           420, 87, 71, 814, 603, 1349, 162, 1027, 783, 326, 101,
                           876, 381, 905, 156, 419, 239, 119, 129, 467])

nht_dead = np.array([-1, 894, 10, 130, 1, 934, 29, 6, 485, 254, 372,
                     81, 191, 355, 180, 286, 47, 269, 361, 173, 246, 492,
                     462, 1319, 58, 297, 31, 2970, 640, 237, 434, 570, 77,
                     271, 563, 3365, 89, 0, 379, 221, 479, 367, 628, 843,
                     1613, 1101, 215, 684, 814, 278, 324, 161, 219, 545, 715,
                     966, 624, 29, 450, 107, 20, 91, 1325, 124, 1468, 104,
                     1309, 429, 62, 1878, 1104, 123, 251, 93, 188, 983, 166,
                     96, 702, 23, 524, 26, 299, 59, 39, 12, 2, 308,
                     1114, 813, 887])

nht_live = np.array([645, 2088, 42, 2090, 11, 886, 1665, 1084, 2900, 2432, 750,
                     4021, 1070, 1765, 1322, 26, 548, 1525, 77, 2181, 2752, 127,
                     2147, 211, 41, 1575, 151, 479, 697, 557, 2267, 542, 392,
                     73, 603, 233, 255, 528, 397, 1529, 1023, 1194, 462, 583,
                     37, 943, 996, 480, 1497, 717, 224, 219, 1531, 498, 44,
                     288, 267, 600, 52, 269, 1086, 386, 176, 2199, 216, 54,
                     675, 1243, 463, 650, 171, 327, 110, 774, 509, 8, 197,
                     136, 12, 1124, 64, 380, 811, 232, 192, 731, 715, 226,
                     605, 539, 1491, 323, 240, 179, 702, 156, 82, 1397, 354,
                     778, 603, 1001, 385, 986, 203, 149, 576, 445, 180, 1403,
                     252, 675, 1351, 2983, 1568, 45, 899, 3260, 1025, 31, 100,
                     2055, 4043, 79, 238, 3931, 2351, 595, 110, 215, 0, 563,
                     206, 660, 242, 577, 179, 157, 192, 192, 1848, 792, 1693,
                     55, 388, 225, 1134, 1172, 1555, 31, 1582, 1044, 378, 1687,
                     2915, 280, 765, 2819, 511, 1521, 745, 2491, 580, 2072, 6450,
                     578, 745, 1075, 1103, 1549, 1520, 138, 1202, 296, 277, 351,
                     391, 950, 459, 62, 1056, 1128, 139, 420, 87, 71, 814,
                     603, 1349, 162, 1027, 783, 326, 101, 876, 381, 905, 156,
                     419, 239, 119, 129, 467])


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


def how_often_do_we_get_no_hitters():
    # Seed random number generator
    np.random.seed(42)

    # Compute mean no-hitter time: tau
    tau = np.mean(nohitter_times)

    # Draw out of an exponential distribution with parameter tau: inter_nohitter_time
    inter_nohitter_time = np.random.exponential(tau, 100000)

    # Plot the PDF and label axes
    _ = plt.hist(inter_nohitter_time, bins=50, density=True, histtype='step')
    _ = plt.xlabel('Games between no-hitters')
    _ = plt.ylabel('PDF')

    # Show the plot
    plt.show()


def does_no_hitter_data_follow_the_story():
    # Seed random number generator
    np.random.seed(42)

    # Compute mean no-hitter time: tau
    tau = np.mean(nohitter_times)

    # Draw out of an exponential distribution with parameter tau: inter_nohitter_time
    inter_nohitter_time = np.random.exponential(tau, 100000)

    # Create an ECDF from real data: x, y
    x, y = ecdf(nohitter_times)

    # Create a CDF from theoretical samples: x_theor, y_theor
    x_theor, y_theor = ecdf(inter_nohitter_time)

    # Overlay the plots
    plt.plot(x_theor, y_theor)
    plt.plot(x, y, marker='.', linestyle='none')

    # Margins and axis labels
    plt.margins(0.02)
    plt.xlabel('Games between no-hitters')
    plt.ylabel('CDF')

    # Show the plot
    plt.show()


def how_is_this_parameter_optimal():
    # Seed random number generator
    np.random.seed(42)

    # Compute mean no-hitter time: tau
    tau = np.mean(nohitter_times)

    # Draw out of an exponential distribution with parameter tau: inter_nohitter_time
    inter_nohitter_time = np.random.exponential(tau, 100000)

    # Create an ECDF from real data: x, y
    x, y = ecdf(nohitter_times)

    # Create a CDF from theoretical samples: x_theor, y_theor
    x_theor, y_theor = ecdf(inter_nohitter_time)

    # Plot the theoretical CDFs
    plt.plot(x_theor, y_theor)
    plt.plot(x, y, marker='.', linestyle='none')
    plt.margins(0.02)
    plt.xlabel('Games between no-hitters')
    plt.ylabel('CDF')

    # Take samples with half tau: samples_half
    samples_half = np.random.exponential(tau / 2, 10000)

    # Take samples with double tau: samples_double
    samples_double = np.random.exponential(tau * 2, 10000)

    # Generate CDFs from these samples
    x_half, y_half = ecdf(samples_half)
    x_double, y_double = ecdf(samples_double)

    # Plot these CDFs as lines
    _ = plt.plot(x_half, y_half)
    _ = plt.plot(x_double, y_double)

    # Show the plot
    plt.show()


def confidence_interval_on_the_rate_of_no_hitters():
    # Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
    bs_replicates = draw_bs_reps(nohitter_times, np.mean, 10000)

    # Compute the 95% confidence interval: conf_int
    conf_int = np.percentile(bs_replicates, [2.5, 97.5])

    # Print the confidence interval
    print('95% confidence interval =', conf_int, 'games')

    # Plot the histogram of the replicates
    _ = plt.hist(bs_replicates, bins=50, density=True)
    _ = plt.xlabel(r'$\tau$ (games)')
    _ = plt.ylabel('PDF')

    # Show the plot
    plt.show()


def a_time_on_website_nohitter_analog():
    # Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
    nht_diff_obs = diff_of_means(nht_dead, nht_live)

    # Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
    perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)

    # Compute and print the p-value: p
    p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
    print('p-val =', p)


def main():
    set_seaborn_styling()
    # generate_randoms_and_plot()
    # perform_bernoulli_trials_example()
    # how_many_defaults_to_expect()
    # will_the_banks_fail()
    # sample_from_binomial_distribution()
    # plot_binomial_pmf()
    # banks_sample_from_binomial_distribution()
    # banks_plot_binomial_pmf()
    # sample_poisson_cdf()
    # show_relationship_between_binomial_and_poisson()
    # was_2015_anomalous()
    # distribution_of_nohitters_and_cycles()
    how_often_do_we_get_no_hitters()
    does_no_hitter_data_follow_the_story()
    how_is_this_parameter_optimal()
    confidence_interval_on_the_rate_of_no_hitters()
    a_time_on_website_nohitter_analog()


if __name__ == '__main__':
    main()
