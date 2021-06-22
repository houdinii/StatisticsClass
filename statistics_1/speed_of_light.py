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


def main():
    set_seaborn_styling()
    michelson_speed_of_light = import_data()
    check_normalty_of_msol(michelson_speed_of_light)
    example_of_normal_pdf()
    example_of_normal_cdf()


if __name__ == '__main__':
    main()
