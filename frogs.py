import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utilities.utilities import \
    ecdf, \
    pearson_r, \
    set_seaborn_styling, \
    perform_bernoulli_trials, \
    successive_poisson, \
    draw_bs_reps, \
    draw_bs_pairs_linreg, \
    draw_perm_reps, \
    permutation_sample, \
    diff_of_means


ID = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
               'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B',
               'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
               'B'])

impact_force = np.array([1.612, 0.605, 0.327, 0.946, 0.541, 1.539, 0.529, 0.628, 1.453,
                        0.297, 0.703, 0.269, 0.751, 0.245, 1.182, 0.515, 0.435, 0.383,
                        0.457, 0.73 , 0.172, 0.142, 0.037, 0.453, 0.355, 0.022, 0.502,
                        0.273, 0.72 , 0.582, 0.198, 0.198, 0.597, 0.516, 0.815, 0.402,
                        0.605, 0.711, 0.614, 0.468])

frogs_data = pd.DataFrame()
frogs_data['ID'] = ID
frogs_data['impact_force'] = impact_force

force_a = np.array([1.612, 0.605, 0.327, 0.946, 0.541, 1.539, 0.529, 0.628, 1.453,
                    0.297, 0.703, 0.269, 0.751, 0.245, 1.182, 0.515, 0.435, 0.383,
                    0.457, 0.730])

force_b = np.array([0.172, 0.142, 0.037, 0.453, 0.355, 0.022, 0.502, 0.273, 0.720,
                    0.582, 0.198, 0.198, 0.597, 0.516, 0.815, 0.402, 0.605, 0.711,
                    0.614, 0.468])

forces_concat = np.array([1.612, 0.605, 0.327, 0.946, 0.541, 1.539, 0.529, 0.628, 1.453,
                         0.297, 0.703, 0.269, 0.751, 0.245, 1.182, 0.515, 0.435, 0.383,
                         0.457, 0.730, 0.172, 0.142, 0.037, 0.453, 0.355, 0.022, 0.502,
                         0.273, 0.720, 0.582, 0.198, 0.198, 0.597, 0.516, 0.815, 0.402,
                         0.605, 0.711, 0.614, 0.468])

empirical_diff_means_two_sample = 0.28825000000000006


def eda_before_hypothesis_testing():
    # Make bee swarm plot
    _ = sns.swarmplot(x='ID', y='impact_force', data=frogs_data)

    # Label axes
    _ = plt.xlabel('frog')
    _ = plt.ylabel('impact force (N)')

    # Show the plot
    plt.show()


def permutation_test_on_frog_data():
    # Compute difference of mean impact force from experiment: empirical_diff_means
    empirical_diff_means = diff_of_means(force_a, force_b)

    # Draw 10,000 permutation replicates: perm_replicates
    perm_replicates = draw_perm_reps(force_a, force_b, diff_of_means, size=10000)

    # Compute p-value: p
    p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

    # Print the result
    print('\n p-value =', p)


def a_one_sample_bootstrap_hypothesis_test():
    # Make an array of translated impact forces: translated_force_b
    translated_force_b = (force_b - np.mean(force_b) + 0.55)

    # Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
    bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

    # Compute fraction of replicates that are less than the observed Frog B force: p
    p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

    # Print the p-value
    print('P-Value After One Sample Bootstrap Test: ', p)


def a_two_sample_bootstrap_hypothesis_test():
    # Compute mean of all forces: mean_force
    mean_force = np.mean(forces_concat)

    # Generate shifted arrays
    force_a_shifted = force_a - np.mean(force_a) + mean_force
    force_b_shifted = force_b - np.mean(force_b) + mean_force

    # Compute 10,000 bootstrap replicates from shifted arrays
    bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
    bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

    # Get replicates of difference of means: bs_replicates
    bs_replicates = bs_replicates_a - bs_replicates_b

    # Compute and print p-value: p
    p = np.sum(bs_replicates >= empirical_diff_means_two_sample) / len(bs_replicates)
    print('p-value in two sample tets =', p)


def main():
    eda_before_hypothesis_testing()
    permutation_test_on_frog_data()
    a_one_sample_bootstrap_hypothesis_test()
    a_two_sample_bootstrap_hypothesis_test()


if __name__ == '__main__':
    main()
