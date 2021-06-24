import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from data_utilities.utilities import ecdf, pearson_r, set_seaborn_styling, perform_bernoulli_trials, \
    bootstrap_replicate_1d, draw_bs_reps, diff_of_means, draw_bs_pairs_linreg

# Import data (Copied and pasted from DataCamp iPython console)
# There is also two data csvs that could be cleaned and extracted from (Exercise for later)
f_year = np.array([1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975,
                   1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 1975, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
                   2012, 2012, 2012, 2012, 2012])
f_beak_depth = np.array([8.4, 8.8, 8.4, 8., 7.9, 8.9, 8.6, 8.5, 8.9,
                         9.1, 8.6, 9.8, 8.2, 9., 9.7, 8.6, 8.2, 9.,
                         8.4, 8.6, 8.9, 9.1, 8.3, 8.7, 9.6, 8.5, 9.1,
                         9., 9.2, 9.9, 8.6, 9.2, 8.4, 8.9, 8.5, 10.4,
                         9.6, 9.1, 9.3, 9.3, 8.8, 8.3, 8.8, 9.1, 10.1,
                         8.9, 9.2, 8.5, 10.2, 10.1, 9.2, 9.7, 9.1, 8.5,
                         8.2, 9., 9.3, 8., 9.1, 8.1, 8.3, 8.7, 8.8,
                         8.6, 8.7, 8., 8.8, 9., 9.1, 9.74, 9.1, 9.8,
                         10.4, 8.3, 9.44, 9.04, 9., 9.05, 9.65, 9.45, 8.65,
                         9.45, 9.45, 9.05, 8.75, 9.45, 8.35, 9.4, 8.9, 9.5,
                         11., 8.7, 8.4, 9.1, 8.7, 10.2, 9.6, 8.85, 8.8,
                         9.5, 9.2, 9., 9.8, 9.3, 9., 10.2, 7.7, 9.,
                         9.5, 9.4, 8., 8.9, 9.4, 9.5, 8., 10., 8.95,
                         8.2, 8.8, 9.2, 9.4, 9.5, 8.1, 9.5, 8.4, 9.3,
                         9.3, 9.6, 9.2, 10., 8.9, 10.5, 8.9, 8.6, 8.8,
                         9.15, 9.5, 9.1, 10.2, 8.4, 10., 10.2, 9.3, 10.8,
                         8.3, 7.8, 9.8, 7.9, 8.9, 7.7, 8.9, 9.4, 9.4,
                         8.5, 8.5, 9.6, 10.2, 8.8, 9.5, 9.3, 9., 9.2,
                         8.7, 9., 9.1, 8.7, 9.4, 9.8, 8.6, 10.6, 9.,
                         9.5, 8.1, 9.3, 9.6, 8.5, 8.2, 8., 9.5, 9.7,
                         9.9, 9.1, 9.5, 9.8, 8.4, 8.3, 9.6, 9.4, 10.,
                         8.9, 9.1, 9.8, 9.3, 9.9, 8.9, 8.5, 10.6, 9.3,
                         8.9, 8.9, 9.7, 9.8, 10.5, 8.4, 10., 9., 8.7,
                         8.8, 8.4, 9.3, 9.8, 8.9, 9.8, 9.1])
df = pd.DataFrame()
df['year'] = f_year
df['beak_depth'] = f_beak_depth

# More copies from iPython console
bd_1975 = np.array([8.4, 8.8, 8.4, 8., 7.9, 8.9, 8.6, 8.5, 8.9,
                    9.1, 8.6, 9.8, 8.2, 9., 9.7, 8.6, 8.2, 9.,
                    8.4, 8.6, 8.9, 9.1, 8.3, 8.7, 9.6, 8.5, 9.1,
                    9., 9.2, 9.9, 8.6, 9.2, 8.4, 8.9, 8.5, 10.4,
                    9.6, 9.1, 9.3, 9.3, 8.8, 8.3, 8.8, 9.1, 10.1,
                    8.9, 9.2, 8.5, 10.2, 10.1, 9.2, 9.7, 9.1, 8.5,
                    8.2, 9., 9.3, 8., 9.1, 8.1, 8.3, 8.7, 8.8,
                    8.6, 8.7, 8., 8.8, 9., 9.1, 9.74, 9.1, 9.8,
                    10.4, 8.3, 9.44, 9.04, 9., 9.05, 9.65, 9.45, 8.65,
                    9.45, 9.45, 9.05, 8.75, 9.45, 8.35])
bd_2012 = np.array([9.4, 8.9, 9.5, 11., 8.7, 8.4, 9.1, 8.7, 10.2,
                    9.6, 8.85, 8.8, 9.5, 9.2, 9., 9.8, 9.3, 9.,
                    10.2, 7.7, 9., 9.5, 9.4, 8., 8.9, 9.4, 9.5,
                    8., 10., 8.95, 8.2, 8.8, 9.2, 9.4, 9.5, 8.1,
                    9.5, 8.4, 9.3, 9.3, 9.6, 9.2, 10., 8.9, 10.5,
                    8.9, 8.6, 8.8, 9.15, 9.5, 9.1, 10.2, 8.4, 10.,
                    10.2, 9.3, 10.8, 8.3, 7.8, 9.8, 7.9, 8.9, 7.7,
                    8.9, 9.4, 9.4, 8.5, 8.5, 9.6, 10.2, 8.8, 9.5,
                    9.3, 9., 9.2, 8.7, 9., 9.1, 8.7, 9.4, 9.8,
                    8.6, 10.6, 9., 9.5, 8.1, 9.3, 9.6, 8.5, 8.2,
                    8., 9.5, 9.7, 9.9, 9.1, 9.5, 9.8, 8.4, 8.3,
                    9.6, 9.4, 10., 8.9, 9.1, 9.8, 9.3, 9.9, 8.9,
                    8.5, 10.6, 9.3, 8.9, 8.9, 9.7, 9.8, 10.5, 8.4,
                    10., 9., 8.7, 8.8, 8.4, 9.3, 9.8, 8.9, 9.8,
                    9.1])
bl_1975 = np.array([13.9, 14., 12.9, 13.5, 12.9, 14.6, 13., 14.2, 14.,
                    14.2, 13.1, 15.1, 13.5, 14.4, 14.9, 12.9, 13., 14.9,
                    14., 13.8, 13., 14.75, 13.7, 13.8, 14., 14.6, 15.2,
                    13.5, 15.1, 15., 12.8, 14.9, 15.3, 13.4, 14.2, 15.1,
                    15.1, 14., 13.6, 14., 14., 13.9, 14., 14.9, 15.6,
                    13.8, 14.4, 12.8, 14.2, 13.4, 14., 14.8, 14.2, 13.5,
                    13.4, 14.6, 13.5, 13.7, 13.9, 13.1, 13.4, 13.8, 13.6,
                    14., 13.5, 12.8, 14., 13.4, 14.9, 15.54, 14.63, 14.73,
                    15.73, 14.83, 15.94, 15.14, 14.23, 14.15, 14.35, 14.95, 13.95,
                    14.05, 14.55, 14.05, 14.45, 15.05, 13.25])
bl_2012 = np.array([14.3, 12.5, 13.7, 13.8, 12., 13., 13., 13.6, 12.8,
                    13.6, 12.95, 13.1, 13.4, 13.9, 12.3, 14., 12.5, 12.3,
                    13.9, 13.1, 12.5, 13.9, 13.7, 12., 14.4, 13.5, 13.8,
                    13., 14.9, 12.5, 12.3, 12.8, 13.4, 13.8, 13.5, 13.5,
                    13.4, 12.3, 14.35, 13.2, 13.8, 14.6, 14.3, 13.8, 13.6,
                    12.9, 13., 13.5, 13.2, 13.7, 13.1, 13.2, 12.6, 13.,
                    13.9, 13.2, 15., 13.37, 11.4, 13.8, 13., 13., 13.1,
                    12.8, 13.3, 13.5, 12.4, 13.1, 14., 13.5, 11.8, 13.7,
                    13.2, 12.2, 13., 13.1, 14.7, 13.7, 13.5, 13.3, 14.1,
                    12.5, 13.7, 14.6, 14.1, 12.9, 13.9, 13.4, 13., 12.7,
                    12.1, 14., 14.9, 13.9, 12.9, 14.6, 14., 13., 12.7,
                    14., 14.1, 14.1, 13., 13.5, 13.4, 13.9, 13.1, 12.9,
                    14., 14., 14.1, 14.7, 13.4, 13.8, 13.4, 13.8, 12.4,
                    14.1, 12.9, 13.9, 14.3, 13.2, 14.2, 13., 14.6, 13.1,
                    15.2])

bs_slope_reps_1975 = None
bs_intercept_reps_1975 = None
bs_slope_reps_2012 = None
bs_intercept_reps_2012 = None

mean_diff = None


def eda_of_beak_depths_of_darwins_finches():
    """
    EDA of beak depths of Darwin's finches
    For your first foray into the Darwin finch data, you will study how the beak depth (the distance, top to bottom,
    of a closed beak) of the finch species Geospiza scandens has changed over time. The Grants have noticed some changes
    of beak geometry depending on the types of seeds available on the island, and they also noticed that there was some
    interbreeding with another major species on Daphne Major, Geospiza fortis. These effects can lead to changes in the
    species over time.

    In the next few problems, you will look at the beak depth of G. scandens on Daphne Major in 1975 and in 2012. To
    start with, let's plot all of the beak depth measurements in 1975 and 2012 in a bee swarm plot.

    The data are stored in a pandas DataFrame called df with columns 'year' and 'beak_depth'. The units of beak depth
    are millimeters (mm).

    Instructions:
    + Create the beeswarm plot.
    + Label the axes.
    + Show the plot.
    """

    # Create bee swarm plot
    _ = sns.swarmplot(x='year', y='beak_depth', data=df)

    # Label the axes
    _ = plt.xlabel('year')
    _ = plt.ylabel('beak depth (mm)')

    # Show the plot
    plt.show()


def ecdfs_of_beak_depths():
    """
    While bee swarm plots are useful, we found that ECDFs are often even better when doing EDA. Plot the ECDFs for the
    1975 and 2012 beak depth measurements on the same plot.

    For your convenience, the beak depths for the respective years has been stored in the NumPy arrays bd_1975 and
    bd_2012.

    Instructions:
    + Compute the ECDF for the 1975 and 2012 data.
    + Plot the two ECDFs.
    + Set a 2% margin and add axis labels and a legend to the plot.
    + Hit 'Submit Answer' to view the plot!
    """

    # Compute ECDFs
    x_1975, y_1975 = ecdf(bd_1975)
    x_2012, y_2012 = ecdf(bd_2012)

    # Plot the ECDFs
    _ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
    _ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

    # Set margins
    plt.margins(0.02)

    # Add axis labels and legend
    _ = plt.xlabel('beak depth (mm)')
    _ = plt.ylabel('ECDF')
    _ = plt.legend(('1975', '2012'), loc='lower right')

    # Show the plot
    plt.show()


def parameter_estimates_of_beak_depths():
    """
    Parameter estimates of beak depths
    Estimate the difference of the mean beak depth of the G. scandens samples from 1975 and 2012 and report a 95%
    confidence interval.

    Since in this exercise you will use the draw_bs_reps() function you wrote in chapter 2, it may be helpful to refer
    back to it.

    Instructions:
    + Compute the difference of the sample means.
    + Take 10,000 bootstrap replicates of the mean for the 1975 beak depths using your draw_bs_reps() function. Also get 10,000 bootstrap replicates of the mean for the 2012 beak depths.
    + Subtract the 1975 replicates from the 2012 replicates to get bootstrap replicates of the difference of means.
    + Use the replicates to compute the 95% confidence interval.
    + Hit 'Submit Answer' to view the results!
    """
    global mean_diff

    # Compute the difference of the sample means: mean_diff and put it in the global namespace
    mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

    # Get bootstrap replicates of means
    bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
    bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)

    # Compute samples of difference of means: bs_diff_replicates
    bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

    # Compute 95% confidence interval: conf_int
    conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])
    # Print the results
    print('Difference of Means =', mean_diff, 'mm')
    print('95% Confidence Interval =', conf_int, 'mm')


def hypothesis_test_are_beaks_deeper_in_2012():
    """
    Hypothesis test: Are beaks deeper in 2012?
    Your plot of the ECDF and determination of the confidence interval make it pretty clear that the beaks of G.
    scandens on Daphne Major have gotten deeper. But is it possible that this effect is just due to random chance?
    In other words, what is the probability that we would get the observed difference in mean beak depth if the means
    were the same?

    Be careful! The hypothesis we are testing is not that the beak depths come from the same distribution. For that we
    could use a permutation test. The hypothesis is that the means are equal. To perform this hypothesis test, we need
    to shift the two data sets so that they have the same mean and then use bootstrap sampling to compute the
    difference of means.

    Instructions:
    + Make a concatenated array of the 1975 and 2012 beak depths and compute and store its mean.
    + Shift bd_1975 and bd_2012 such that their means are equal to the one you just computed for the combined data set.
    + Take 10,000 bootstrap replicates of the mean each for the 1975 and 2012 beak depths.
    + Subtract the 1975 replicates from the 2012 replicates to get bootstrap replicates of the difference.
    + Compute and print the p-value. The observed difference in means you computed in the last exercise is still in your
      namespace as mean_diff.
    """

    # Compute mean of combined data set: combined_mean
    combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

    # Shift the samples
    bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
    bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

    # Get bootstrap replicates of shifted data sets
    bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, 10000)
    bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, 10000)

    # Compute replicates of difference of means: bs_diff_replicates
    bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

    # Compute the p-value
    # p = np.sum(bs_replicates >= empirical_diff_means_two_sample) / len(bs_replicates)
    p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

    # Print p-value
    print('P-Value =', p)
    """
    We get a p-value of 0.0034, which suggests that there is a statistically significant difference. 
    But remember: it is very important to know how different they are! In the previous exercise, you got a difference 
    of 0.2 mm between the means. You should combine this with the statistical significance. Changing by 0.2 mm in 37 
    years is substantial by evolutionary standards. If it kept changing at that rate, the beak depth would double in 
    only 400 years.
    """


def eda_of_beak_length_and_depth():
    # Make scatter plot of 1975 data
    _ = plt.plot(bl_1975, bd_1975, marker='.', linestyle='None', color='blue', alpha=0.5)

    # Make scatter plot of 2012 data
    _ = plt.plot(bl_2012, bd_2012, marker='.', linestyle='None', color='red', alpha=0.5)

    # Label axes and make legend
    _ = plt.xlabel('beak length (mm)')
    _ = plt.ylabel('beak depth (mm)')
    _ = plt.legend(('1975', '2012'), loc='upper left')

    # Show the plot
    plt.show()

    """
    Great work! In looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction), 
    but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. 
    So, it does not look like the beaks kept the same shape; they became shorter and deeper.
    """


def linear_regressions():
    """
    Linear regressions
    Perform a linear regression for both the 1975 and 2012 data. Then, perform pairs bootstrap estimates for the
    regression parameters. Report 95% confidence intervals on the slope and intercept of the regression line.

    You will use the draw_bs_pairs_linreg() function you wrote back in chapter 2.

    As a reminder, its call signature is draw_bs_pairs_linreg(x, y, size=1), and it returns bs_slope_reps and
    bs_intercept_reps. The beak length data are stored as bl_1975 and bl_2012, and the beak depth data is stored
    in bd_1975 and bd_2012.

    Instructions:
    + Compute the slope and intercept for both the 1975 and 2012 data sets.
    + Obtain 1000 pairs bootstrap samples for the linear regressions using your draw_bs_pairs_linreg() function.
    + Compute 95% confidence intervals for the slopes and the intercepts.
    """
    global bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, bs_intercept_reps_2012

    # Compute the linear regressions
    [slope_1975, intercept_1975] = np.polyfit(bl_1975, bd_1975, 1)
    [slope_2012, intercept_2012] = np.polyfit(bl_2012, bd_2012, 1)

    # Perform pairs bootstrap for the linear regressions
    bs_slope_reps_1975, bs_intercept_reps_1975 = draw_bs_pairs_linreg(bl_1975, bd_1975, 1000)
    bs_slope_reps_2012, bs_intercept_reps_2012 = draw_bs_pairs_linreg(bl_2012, bd_2012, 1000)

    # Compute confidence intervals of slopes
    slope_conf_int_1975 = np.percentile(bs_slope_reps_1975, [2.5, 97.5])
    slope_conf_int_2012 = np.percentile(bs_slope_reps_2012, [2.5, 97.5])
    intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975, [2.5, 97.5])
    intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012, [2.5, 97.5])

    # Print the results
    print('1975: slope =', slope_1975, 'conf int =', slope_conf_int_1975)
    print('1975: intercept =', intercept_1975, 'conf int =', intercept_conf_int_1975)
    print('2012: slope =', slope_2012, 'conf int =', slope_conf_int_2012)
    print('2012: intercept =', intercept_2012, 'conf int =', intercept_conf_int_2012)

    """Nicely done! It looks like they have the same slope, but different intercepts."""


def displaying_the_linear_regression_results():
    """
    Displaying the linear regression results
    Now, you will display your linear regression results on the scatter plot, the code for which is already pre-written
    for you from your previous exercise. To do this, take the first 100 bootstrap samples (stored in bs_slope_reps_1975,
    bs_intercept_reps_1975, bs_slope_reps_2012, and bs_intercept_reps_2012) and plot the lines with alpha=0.2 and
    linewidth=0.5 keyword arguments to plt.plot().

    Instructions:
    + Generate the x-values for the bootstrap lines using np.array(). They should consist of 10 mm and 17 mm.
    + Write a for loop to plot 100 of the bootstrap lines for the 1975 and 2012 data sets. The lines for the 1975 data
      set should be 'blue' and those for the 2012 data set should be 'red'.
    + Hit 'Submit Answer' to view the plot!
    """
    global bs_slope_reps_1975, bs_intercept_reps_1975, bs_slope_reps_2012, bs_intercept_reps_2012

    # Make scatter plot of 1975 data
    _ = plt.plot(bl_1975, bd_1975, marker='.', linestyle='none', color='blue', alpha=0.5)

    # Make scatter plot of 2012 data
    _ = plt.plot(bl_2012, bd_2012, marker='.', linestyle='none', color='red', alpha=0.5)

    # Label axes and make legend
    _ = plt.xlabel('beak length (mm)')
    _ = plt.ylabel('beak depth (mm)')
    _ = plt.legend(('1975', '2012'), loc='upper left')

    # Generate x-values for bootstrap lines: x
    x = np.array([10, 17])

    # Plot the bootstrap lines
    for i in range(100):
        plt.plot(x, bs_slope_reps_1975[i] * x + bs_intercept_reps_1975[i], linewidth=0.5, alpha=0.2, color='blue')
        plt.plot(x, bs_slope_reps_2012[i] * x + bs_intercept_reps_2012[i], linewidth=0.5, alpha=0.2, color='red')

    # Draw the plot again
    plt.show()


def main():
    set_seaborn_styling()
    eda_of_beak_depths_of_darwins_finches()
    ecdfs_of_beak_depths()
    parameter_estimates_of_beak_depths()
    hypothesis_test_are_beaks_deeper_in_2012()
    eda_of_beak_length_and_depth()
    linear_regressions()
    displaying_the_linear_regression_results()


if __name__ == '__main__':
    main()
