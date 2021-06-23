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
    draw_bs_reps, \
    draw_bs_pairs_linreg

illiteracy = np.array([9.5, 49.2, 1, 11.2, 9.8, 60, 50.2, 51.2, 0.6, 1, 8.5,
                       6.1, 9.8, 1, 42.2, 77.2, 18.7, 22.8, 8.5, 43.9, 1, 1,
                       1.5, 10.8, 11.9, 3.4, 0.4, 3.1, 6.6, 33.7, 40.4, 2.3, 17.2,
                       0.7, 36.1, 1, 33.2, 55.9, 30.8, 87.4, 15.4, 54.6, 5.1, 1.1,
                       10.2, 19.8, 0, 40.7, 57.2, 59.9, 3.1, 55.7, 22.8, 10.9, 34.7,
                       32.2, 43, 1.3, 1, 0.5, 78.4, 34.2, 84.9, 29.1, 31.3, 18.3,
                       81.8, 39, 11.2, 67, 4.1, 0.2, 78.1, 1, 7.1, 1, 29,
                       1.1, 11.7, 73.6, 33.9, 14, 0.3, 1, 0.8, 71.9, 40.1, 1,
                       2.1, 3.8, 16.5, 4.1, 0.5, 44.4, 46.3, 18.7, 6.5, 36.8, 18.6,
                       11.1, 22.1, 71.1, 1, 0, 0.9, 0.7, 45.5, 8.4, 0, 3.8,
                       8.5, 2, 1, 58.9, 0.3, 1, 14, 47, 4.1, 2.2, 7.2,
                       0.3, 1.5, 50.5, 1.3, 0.6, 19.1, 6.9, 9.2, 2.2, 0.2, 12.3,
                       4.9, 4.6, 0.3, 16.5, 65.7, 63.5, 16.8, 0.2, 1.8, 9.6, 15.2,
                       14.4, 3.3, 10.6, 61.3, 10.9, 32.2, 9.3, 11.6, 20.7, 6.5, 6.7,
                       3.5, 1, 1.6, 20.5, 1.5, 16.7, 2, 0.9])

fertility = np.array([1.769, 2.682, 2.077, 2.132, 1.827, 3.872, 2.288, 5.173, 1.393,
                      1.262, 2.156, 3.026, 2.033, 1.324, 2.816, 5.211, 2.1, 1.781,
                      1.822, 5.908, 1.881, 1.852, 1.3, 2.281, 2.505, 1.224, 1.361,
                      1.468, 2.404, 5.5, 4.058, 2.223, 4.859, 1.267, 2.342, 1.579,
                      6.254, 2.334, 3.961, 6.505, 2.5, 2.823, 2.498, 2.248, 2.508,
                      3.0, 1.854, 4.2, 5.1, 4.967, 1.325, 4.514, 3.173, 2.308,
                      4.6, 4.541, 5.637, 1.926, 1.747, 2.294, 5.841, 5.455, 7.069,
                      2.859, 4.018, 2.513, 5.405, 5.737, 3.363, 4.8, 1.385, 1.505,
                      6.081, 1.784, 1.378, 1.4, 1.841, 1.3, 2.612, 5.329, 5.3,
                      3.371, 1.281, 1.871, 2.153, 5.378, 4.4, 1.4, 1.436, 1.612,
                      3.1, 2.752, 3.3, 4.0, 4.166, 2.642, 2.977, 3.415, 2.295,
                      3.019, 2.683, 5.165, 1.849, 1.836, 2.518, 2.4, 4.528, 1.263,
                      1.885, 1.943, 1.899, 1.442, 1.953, 4.697, 1.582, 2.025, 1.841,
                      5.011, 1.212, 1.502, 2.516, 1.367, 2.089, 4.388, 1.854, 1.748,
                      2.978, 2.152, 2.362, 1.988, 1.426, 3.2, 3.264, 1.436, 1.393,
                      2.822, 4.969, 5.659, 3.2, 1.693, 1.647, 2.3, 1.792, 3.4,
                      1.516, 2.233, 2.563, 5.283, 3.885, 0.966, 2.373, 2.663, 1.251,
                      2.052, 3.371, 2.093, 2, 3.883, 3.852, 3.718, 1.732, 3.928])


def eda_of_literacy_fertility_data():
    # Plot the illiteracy rate versus fertility
    _ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

    # Set the margins and label axes
    plt.margins(0.02)
    _ = plt.xlabel('percent illiterate')
    _ = plt.ylabel('fertility')

    # Show the plot
    plt.show()

    # Show the Pearson correlation coefficient
    p_r = pearson_r(illiteracy, fertility)
    print(f"Pearson Correlation Coefficient: {p_r}\n")
    return p_r


def linear_regression():
    # Plot the illiteracy rate versus fertility
    _ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
    plt.margins(0.02)
    _ = plt.xlabel('percent illiterate')
    _ = plt.ylabel('fertility')

    # Perform a linear regression using np.polyfit(): a, b
    [a, b] = np.polyfit(illiteracy, fertility, 1)

    # Print the results to the screen
    print('slope =', a, 'children per woman / percent illiterate')
    print('intercept =', b, 'children per woman')

    # Make theoretical line to plot
    x = np.array([0, 100])
    y = a * x + b

    # Add regression line to your plot
    _ = plt.plot(x, y)

    # Draw the plot
    plt.show()
    return a, b


def how_is_it_optimal(a, b):
    # Specify slopes to consider: a_vals
    a_vals = np.linspace(0, 0.1, 200)

    # Initialize sum of square of residuals: rss
    rss = np.empty_like(a_vals)

    # Compute sum of square of residuals for each value of a_vals
    for i, a in enumerate(a_vals):
        rss[i] = np.sum((fertility - a * illiteracy - b) ** 2)

    # Plot the RSS
    plt.plot(a_vals, rss, '-')
    plt.xlabel('slope (children per woman / percent illiterate)')
    plt.ylabel('sum of square of residuals')

    plt.show()


def linear_regression_on_appropriate_anscombe_data():
    x = np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.])
    y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    # Perform linear regression: a, b
    [a, b] = np.polyfit(x, y, 1)

    # Print the slope and intercept
    print(a, b)

    # Generate theoretical x and y data: x_theor, y_theor
    x_theor = np.array([3, 15])
    y_theor = x_theor * a + b

    # Plot the Anscombe data and theoretical line
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.plot(x_theor, y_theor)

    # Label the axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.show()


def linear_regression_on_all_anscombe_data():
    anscombe_x = [np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.]),
                  np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.]),
                  np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.]),
                  np.array([8., 8., 8., 8., 8., 8., 8., 19., 8., 8., 8.])]

    anscombe_y = [np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
                  np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74]),
                  np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
                  np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89])]

    # Iterate through x,y pairs
    print('\n')
    for x, y in zip(anscombe_x, anscombe_y):
        # Compute the slope and intercept: a, b
        [a, b] = np.polyfit(x, y, 1)

        # Print the result
        print('slope:', a, 'intercept:', b)
    print('\n')


def pairs_bootstrap_of_literacy_fertility_data():
    # Generate replicates of slope and intercept using pairs bootstrap
    bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, 1000)

    # Compute and print 95% CI for slope
    print(f"\n95% Confidence Interval: {np.percentile(bs_slope_reps, [2.5, 97.5])}\n")

    # Plot the histogram
    _ = plt.hist(bs_slope_reps, bins=50, density=True)
    _ = plt.xlabel('slope')
    _ = plt.ylabel('PDF')
    plt.show()


def hypothesis_test_on_pearson_correlation():
    # Compute observed correlation: r_obs
    r_obs = pearson_r(illiteracy, fertility)

    # Initialize permutation replicates: perm_replicates
    perm_replicates = np.empty(10000)

    # Draw replicates
    for i in range(10000):
        # Permute illiteracy measurments: illiteracy_permuted
        illiteracy_permuted = np.random.permutation(illiteracy)

        # Compute Pearson correlation
        perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

    # Compute p-value: p
    p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
    print('p-val =', p)


def main():
    set_seaborn_styling()
    pearson_co = eda_of_literacy_fertility_data()
    slope, intercept = linear_regression()
    how_is_it_optimal(slope, intercept)
    linear_regression_on_appropriate_anscombe_data()
    linear_regression_on_all_anscombe_data()
    pairs_bootstrap_of_literacy_fertility_data()
    hypothesis_test_on_pearson_correlation()


if __name__ == '__main__':
    main()
