import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utilities.utilities import draw_perm_reps


def exploratory_data_analysis():
    # Exploratory Data Analysis
    df_swing = pd.read_csv('data/2008_swing_states.csv')
    df_all = pd.read_csv('data/2008_all_states.csv')
    print(df_swing[['state', 'county', 'dem_share']])
    return df_swing, df_all


def set_seaborn_styling():
    sns.set()


def generate_swarm_plot(df_swing):
    _ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
    _ = plt.xlabel('state')
    _ = plt.ylabel('percent of vote for Obama')
    plt.show()


def generate_histogram(df_swing):
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    _ = plt.hist(df_swing['dem_share'], bins=bin_edges)  # you can also use a number like 10
    _ = plt.xlabel('percent of vote for Obama')
    _ = plt.ylabel('number of counties')
    plt.show()


def make_ecdf(df_swing):
    x = np.sort(df_swing['dem_share'])
    y = np.arange(1, len(x) + 1) / len(x)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('percent of vote for Obama')
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)  # Keeps data off plot edges
    plt.show()


def make_split_ecdf(df_swing=pd.DataFrame()):
    pa_df = df_swing[(df_swing.state == "PA")]
    oh_df = df_swing[(df_swing.state == "OH")]
    fl_df = df_swing[(df_swing.state == "FL")]
    pa_x = np.sort(pa_df['dem_share'])
    pa_y = np.arange(1, len(pa_x) + 1) / len(pa_x)
    oh_x = np.sort(oh_df['dem_share'])
    oh_y = np.arange(1, len(oh_x) + 1) / len(oh_x)
    fl_x = np.sort(fl_df['dem_share'])
    fl_y = np.arange(1, len(fl_x) + 1) / len(fl_x)
    _ = plt.plot(pa_x, pa_y, marker='.', linestyle='none')
    _ = plt.plot(oh_x, oh_y, marker='.', linestyle='none')
    _ = plt.plot(fl_x, fl_y, marker='.', linestyle='none')
    _ = plt.xlabel('percent of vote for Obama')
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)  # Keeps data off plot edges
    plt.show()


def get_shares(df_swing=pd.DataFrame()):
    pa_df = df_swing[(df_swing.state == "PA")]
    oh_df = df_swing[(df_swing.state == "OH")]
    fl_df = df_swing[(df_swing.state == "FL")]
    return pa_df['dem_share'], oh_df['dem_share'], fl_df['dem_share']


def get_mean_vote_percentage(dem_share_PA, dem_share_OH, dem_share_FL):
    pa = np.mean(dem_share_PA)
    oh = np.mean(dem_share_OH)
    fl = np.mean(dem_share_FL)

    print("")
    print("Mean Vote Percentages of Dem Share")
    print(f"PA: {pa}")
    print(f"OH: {oh}")
    print(f"FL: {fl}")
    return pa, oh, fl


def compute_initial_percentiles(df_swing=pd.DataFrame()):
    percentile = np.percentile(df_swing['dem_share'], [25, 50, 75])
    print(f"Initial percentiles: {percentile}\n")


def generate_box_plot_all(df_all):
    _ = sns.boxplot(x='east_west', y='dem_share', data=df_all)
    _ = plt.xlabel('region')
    _ = plt.ylabel('percent of vote for Obama')
    plt.show()


def compute_variance(dem_share_PA, dem_share_OH, dem_share_FL):
    pa = np.var(dem_share_PA)
    oh = np.var(dem_share_OH)
    fl = np.var(dem_share_FL)
    print("Variance Calculations:")
    print(f"PA: {pa}")
    print(f"OH: {oh}")
    print(f"FL: {fl}\n")
    return pa, oh, fl


def compute_stds(dem_share_PA, dem_share_OH, dem_share_FL):
    pa = np.std(dem_share_PA)
    oh = np.std(dem_share_OH)
    fl = np.std(dem_share_FL)
    print("Standard Deviation Calculations:")
    print(f"PA: {pa}")
    print(f"OH: {oh}")
    print(f"FL: {fl}\n")
    return pa, oh, fl


def generate_scatter_plot(df_all):
    total_votes = df_all['total_votes']
    dem_share = df_all['dem_share']
    _ = plt.plot(total_votes/1000, dem_share, marker='.', linestyle='none')
    _ = plt.xlabel('total votes (thousands)')
    _ = plt.ylabel('percent of vote for Obama')
    plt.show()


def simulate_coin_flips():
    np.random.seed(42)
    random_numbers = np.random.random(size=4)
    print(f"Random Numbers: {random_numbers}")
    heads = random_numbers < 0.5
    print(f"Heads (True/False): {heads}")
    print(f"Total # Of Heads: {np.sum(heads)}\n")


def simulate_trials_of_coin_flips():
    n_all_heads = 0
    for _ in range(10000):
        heads = np.random.random(size=4) < 0.5
        n_heads = np.sum(heads)
        if n_heads == 4:
            n_all_heads += 1
    print(f"Percent of trials with all four heads: {n_all_heads / 10000}\n")


def least_squares_with_np_polyfit_example(total_votes, dem_share):
    [slope, intercept] = np.polyfit(total_votes, dem_share, 1)
    print(f"\nSlope: {slope}")
    print(f"Intercept: {intercept}\n")


def generating_a_pairs_bootstrap_sample(total_votes, dem_share):
    inds = np.arange(len(total_votes))
    bs_inds = np.random.choice(inds, len(inds))
    bs_total_votes = total_votes[bs_inds]
    bs_dem_share = dem_share[bs_inds]

    [bs_slope, bs_intercept] = np.polyfit(bs_total_votes, bs_dem_share, 1)
    [orig_slope, orig_intercept] = np.polyfit(total_votes, dem_share, 1)

    print(f"\nOriginal     Slope: {orig_slope}    Intercept: {orig_intercept}")
    print(f"Bootstrap    Slope: {bs_slope}     Intercept: {bs_intercept}\n")


def generating_a_permutation_sample(dem_share_PA, dem_share_OH):
    dem_share_both = np.concatenate((dem_share_PA, dem_share_OH))
    dem_share_perm = np.random.permutation(dem_share_both)
    perm_sample_PA = dem_share_perm[:len(dem_share_PA)]
    perm_sample_OH = dem_share_perm[len(dem_share_OH):]

    # The Permutation Replicate:
    print(f"Difference In Means, Perm Sample: {np.mean(perm_sample_PA) - np.mean(perm_sample_OH)}")
    print(f"Difference In Means, Original Sample: {np.mean(dem_share_PA) - np.mean(dem_share_OH)}")


def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems[True]) / len(dems)
    return frac


def the_vote_for_the_civil_rights_act_in_1964():
    # Construct arrays of data: dems, reps
    dems = np.array([True] * 153 + [False] * 91)
    reps = np.array([True] * 136 + [False] * 35)

    # Acquire permutation samples: perm_replicates
    perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, 10000)

    # Compute and print p-value: p
    p = np.sum(perm_replicates <= 153 / 244) / len(perm_replicates)
    print('p-value =', p)


def main():
    df_swing, df_all = exploratory_data_analysis()
    set_seaborn_styling()
    # generate_histogram(df_swing)
    # generate_swarm_plot(df_swing)
    # make_ecdf(df_swing)
    # make_split_ecdf(df_swing)

    dem_share_PA, dem_share_OH, dem_share_FL = get_shares(df_swing)
    # mean_vote_percentage = get_mean_vote_percentage(dem_share_PA, dem_share_OH, dem_share_FL)
    # compute_initial_percentiles(df_swing)
    # generate_box_plot_all(df_all)

    # pa_var, oh_var, fl_var = compute_variance(dem_share_PA, dem_share_OH, dem_share_FL)
    # pa_std, oh_std, fl_std = compute_stds(dem_share_PA, dem_share_OH, dem_share_FL)

    # generate_scatter_plot(df_swing)

    # simulate_coin_flips()
    # simulate_trials_of_coin_flips()

    least_squares_with_np_polyfit_example(df_swing['total_votes'], df_swing['dem_share'])
    generating_a_pairs_bootstrap_sample(df_swing['total_votes'], df_swing['dem_share'])
    generating_a_permutation_sample(dem_share_PA, dem_share_OH)

    the_vote_for_the_civil_rights_act_in_1964()


if __name__ == '__main__':
    main()
