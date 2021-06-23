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
    bootstrap_replicate_1d, \
    draw_bs_reps

NEWCOMB_VALUE = 299860  # km/s


def import_data():
    # Exploratory Data Analysis
    df = pd.read_csv('data/michelson_speed_of_light.csv')
    return df['velocity of light in air (km/s)']


def check_normalty_of_msol(michelson_speed_of_light):
    mean = np.mean(michelson_speed_of_light)
    std = np.std(michelson_speed_of_light)
    samples = np.random.normal(mean, std, size=10000)
    x, y = ecdf(michelson_speed_of_light)
    x_theor, y_theor = ecdf(samples)
    _ = plt.plot(x_theor, y_theor)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('speed of light (km/s)')
    _ = plt.ylabel('CDF')
    plt.show()


def example_of_normal_pdf():
    # Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
    samples_std1 = np.random.normal(20, 1, size=100000)
    samples_std3 = np.random.normal(20, 3, size=100000)
    samples_std10 = np.random.normal(20, 10, size=100000)

    # Make histograms
    _ = plt.hist(samples_std1, bins=100, density=True, histtype='step')
    _ = plt.hist(samples_std3, bins=100, density=True, histtype='step')
    _ = plt.hist(samples_std10, bins=100, density=True, histtype='step')

    # Make a legend, set limits and show plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
    plt.ylim(-0.01, 0.42)
    plt.show()


def example_of_normal_cdf():
    # Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
    samples_std1 = np.random.normal(20, 1, size=100000)
    samples_std3 = np.random.normal(20, 3, size=100000)
    samples_std10 = np.random.normal(20, 10, size=100000)

    # Generate CDFs
    x_std1, y_std1 = ecdf(samples_std1)
    x_std3, y_std3 = ecdf(samples_std3)
    x_std10, y_std10 = ecdf(samples_std10)

    # Plot CDFs
    _ = plt.plot(x_std1, y_std1, marker=".", linestyle='none')
    _ = plt.plot(x_std3, y_std3, marker=".", linestyle='none')
    _ = plt.plot(x_std10, y_std10, marker=".", linestyle='none')

    # Make a legend and show the plot
    _ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
    plt.show()


def computing_a_bootstrap_replicate(msol):
    bs_sample = np.random.choice(msol, size=100)
    print(f"\nMean: {np.mean(bs_sample)}")
    print(f"Median: {np.median(bs_sample)}")
    print(f"Standard Deviation: {np.std(bs_sample)}\n")


def bootstrap_replicate_function_example(michelson_speed_of_light):
    print(f"\nExample #1: {bootstrap_replicate_1d(michelson_speed_of_light, np.mean)}")
    print(f"Example #2: {bootstrap_replicate_1d(michelson_speed_of_light, np.mean)}")
    print(f"Example #3: {bootstrap_replicate_1d(michelson_speed_of_light, np.mean)}")
    print(f"Example #4: {bootstrap_replicate_1d(michelson_speed_of_light, np.mean)}")
    print(f"Example #5: {bootstrap_replicate_1d(michelson_speed_of_light, np.mean)}\n")


def many_bootstrap_replicates(michelson_speed_of_light):
    bs_replicates = np.empty(10000)
    for i in range(10000):
        bs_replicates[i] = bootstrap_replicate_1d(michelson_speed_of_light, np.mean)
    _ = plt.hist(bs_replicates, bins=30, density=True)
    _ = plt.xlabel('mean speed of light (km/s)')
    _ = plt.ylabel('PDF')
    plt.show()
    return bs_replicates


def bootstrap_confidence_interval(bs_replicates):
    conf_int = np.percentile(bs_replicates, [2.5, 97.5])
    print(f"\nBootstrap Confidence Interval: {conf_int}\n")


def shifting_the_michelson_data(michelson_speed_of_light):
    michelson_shfted = michelson_speed_of_light - np.mean(michelson_speed_of_light) + NEWCOMB_VALUE
    return michelson_shfted


def diff_from_newcomb(data, newcomb_value=NEWCOMB_VALUE):
    return np.mean(data) - newcomb_value


def calculating_the_test_statistic(michelson_speed_of_light):
    diff_obs = diff_from_newcomb(michelson_speed_of_light)
    print(f"Diff_Obs: {diff_obs}")
    return diff_obs


def computing_the_p_value(michelson_shifted, func, size, diff_observed):
    bs_replicates = draw_bs_reps(michelson_shifted, func, size)
    p_value = np.sum(bs_replicates <= diff_observed) / 10000
    print(f"P-Value: {p_value}\n")
    return p_value


def main():
    set_seaborn_styling()
    michelson_speed_of_light = import_data()
    check_normalty_of_msol(michelson_speed_of_light)
    example_of_normal_pdf()
    example_of_normal_cdf()
    computing_a_bootstrap_replicate(michelson_speed_of_light)
    bootstrap_replicate_function_example(michelson_speed_of_light)
    bs_replicates = many_bootstrap_replicates(michelson_speed_of_light)
    bootstrap_confidence_interval(bs_replicates)
    michelson_shifted = shifting_the_michelson_data(michelson_speed_of_light)
    diff_obs = calculating_the_test_statistic(michelson_speed_of_light)
    p_value = computing_the_p_value(michelson_shifted, diff_from_newcomb, 10000, diff_obs)


if __name__ == '__main__':
    main()
