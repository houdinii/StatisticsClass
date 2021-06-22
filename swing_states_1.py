import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_data_analysis():
    # Exploratory Data Analysis
    df_swing = pd.read_csv('data/2008_swing_states.csv')
    print(df_swing[['state', 'county', 'dem_share']])
    return df_swing


def set_seaborn_styling():
    sns.set()


def generate_histogram(df_swing):
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    _ = plt.hist(df_swing['dem_share'], bins=bin_edges)  # you can also use a number like 10
    _ = plt.xlabel('percent of vote for Obama')
    _ = plt.ylabel('number of counties')
    plt.show()


def main():
    df_swing = exploratory_data_analysis()
    set_seaborn_styling()
    generate_histogram(df_swing)


if __name__ == '__main__':
    main()
