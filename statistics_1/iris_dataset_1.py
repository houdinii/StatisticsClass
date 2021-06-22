import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from data_utilities.utilities import ecdf, pearson_r, set_seaborn_styling


def eda():
    # Import dataset
    iris_ds = load_iris()

    print(iris_ds.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    print(iris_ds['feature_names'])
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print(iris_ds['target_names'])
    # ['setosa' 'versicolor' 'virginica']

    return iris_ds


def get_setosa_petal_length(data, targets):
    result_list = []
    for i in range(len(targets)):
        if targets[i] == 0:
            result_list.append(data[i][2])
    return np.array(result_list)


def get_versicolor_petal_length(data, targets):
    result_list = []
    for i in range(len(targets)):
        if targets[i] == 1:
            result_list.append(data[i][2])
    return np.array(result_list)


def get_versicolor_petal_width(data, targets):
    result_list = []
    for i in range(len(targets)):
        if targets[i] == 1:
            result_list.append(data[i][3])
    return np.array(result_list)


def get_virginica_petal_length(data, targets):
    result_list = []
    for i in range(len(targets)):
        if targets[i] == 2:
            result_list.append(data[i][2])
    return np.array(result_list)


def plot_versicolor_petal_length(versicolor_petal_length):
    # Compute number of data points: n_data
    # Number of bins is the square root of number of data points: n_bins
    # Convert number of bins to integer: n_bins
    n_data = len(versicolor_petal_length)
    n_bins = np.sqrt(n_data)
    n_bins = int(n_bins)

    plt.hist(versicolor_petal_length, bins=n_bins)
    plt.xlabel('petal length (cm)')
    plt.ylabel('count')
    plt.show()


def plot_swarm(iris_df):
    # Create bee swarm plot with Seaborn's default settings
    sns.swarmplot(x='species', y='petal length (cm)', data=iris_df)
    plt.xlabel('Species')
    plt.ylabel('Petal Length')
    plt.show()


def get_all_dataframe(iris_data, iris_target):
    results = []
    for i in range(len(iris_data)):
        species = ""
        if iris_target[i] == 0:
            species = "setosa"
        elif iris_target[i] == 1:
            species = "versicolor"
        elif iris_target[i] == 2:
            species = "virginica"
        results.append([iris_data[i][0], iris_data[i][1], iris_data[i][2], iris_data[i][3], species])
    return pd.DataFrame(results, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species'])


def plot_initial_ecdf(versicolor_petal_length):
    # Compute ECDF for versicolor data: x_vers, y_vers
    x_vers, y_vers = ecdf(versicolor_petal_length)
    _ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
    _ = plt.xlabel('Versicolor Petal Length')
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)
    plt.show()


def plot_comparison_of_ecdfs(setosa_petal_length, versicolor_petal_length, virginica_petal_length):
    # Compute ECDFs
    x_set, y_set = ecdf(setosa_petal_length)
    x_vers, y_vers = ecdf(versicolor_petal_length)
    x_virg, y_virg = ecdf(virginica_petal_length)

    # Plot all ECDFs on the same plot
    _ = plt.plot(x_set, y_set, marker=".", linestyle='none')
    _ = plt.plot(x_vers, y_vers, marker=".", linestyle='none')
    _ = plt.plot(x_virg, y_virg, marker=".", linestyle='none')

    # Annotate the plot
    plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
    _ = plt.xlabel('petal length (cm)')
    _ = plt.ylabel('ECDF')

    # Display the plot
    plt.show()


def compute_means(setosa_petal_length, versicolor_petal_length, virginica_petal_length):
    setosa = np.mean(setosa_petal_length)
    versicolor = np.mean(versicolor_petal_length)
    virginica = np.mean(virginica_petal_length)
    print("")
    print("Mean Lengths:")
    print(f"Setosa: {setosa}")
    print(f"Versicolor: {versicolor}")
    print(f"Virginica: {virginica}\n")
    return setosa, versicolor, virginica


def compute_percentiles(setosa_petal_length, versicolor_petal_length, virginica_petal_length):
    percentiles = np.array([2.5, 25, 50, 75, 97.5])
    setosa = np.percentile(setosa_petal_length, percentiles)
    versicolor = np.percentile(versicolor_petal_length, percentiles)
    virginica = np.percentile(virginica_petal_length, percentiles)
    print("Percentiles")
    print(f"Setosa: {setosa}")
    print(f"Versicolor: {versicolor}")
    print(f"Virginica: {virginica}\n")
    return setosa, versicolor, virginica


def compare_versicolor_ecdf_to_percentiles(versicolor_petal_length):
    percentiles = np.array([2.5, 25, 50, 75, 97.5])
    x_vers, y_vers = ecdf(versicolor_petal_length)
    _ = plt.plot(x_vers, y_vers, '.')
    _ = plt.xlabel('petal length (cm)')
    _ = plt.ylabel('ECDF')
    ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
    _ = plt.plot(ptiles_vers, percentiles / 100, marker='D', color='red', linestyle='none')
    plt.show()


def generate_box_and_whisker(iris_df):
    _ = sns.boxplot(x='species', y='petal length (cm)', data=iris_df)
    _ = plt.xlabel('Species')
    _ = plt.ylabel('Petal Length')
    plt.show()


def compute_explicit_and_calculated_variances(versicolor_petal_length):
    # Array of differences to mean: differences
    differences = versicolor_petal_length - np.mean(versicolor_petal_length)

    # Square the differences: diff_sq
    diff_sq = differences ** 2

    # Compute the mean square difference: variance_explicit
    variance_explicit = np.mean(diff_sq)

    # Compute the variance using NumPy: variance_np
    variance_np = np.var(versicolor_petal_length)

    # Print the results
    print("Computing The Variance: ")
    print(f"variance_explicit: {variance_explicit}")
    print(f"variance_np: {variance_np}\n")
    return variance_np


def compute_explicit_and_calculated_stds(versicolor_petal_length):
    # Compute the variance: variance
    variance = np.var(versicolor_petal_length)

    # Print the square root of the variance
    std_explicit = np.sqrt(variance)

    # Print the standard deviation
    std_calculated = np.std(versicolor_petal_length)

    # Print the results
    print("Computing The Standard Deviations: ")
    print(f"std_explicit: {std_explicit}")
    print(f"std_np: {std_calculated}\n")
    return std_calculated


def generate_scatter_plot(versicolor_petal_length, versicolor_petal_width):
    _ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
    _ = plt.xlabel('Petal Length')
    _ = plt.ylabel('Petal Width')
    plt.show()


def calculate_covariance(versicolor_petal_length, versicolor_petal_width):
    # Compute the covariance matrix: covariance_matrix
    covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

    # Print covariance matrix
    print(f"covariance_matrix:\n{covariance_matrix}\n")

    # Extract covariance of length and width of petals: petal_cov
    petal_cov = covariance_matrix[0, 1]

    # Print the length/width covariance
    print(f"Covariance of Length and Width: {petal_cov}\n")
    return covariance_matrix, petal_cov


def calculate_pearson_r(versicolor_petal_length, versicolor_petal_width):
    # Compute Pearson correlation coefficient for I. versicolor: r
    r = pearson_r(versicolor_petal_length, versicolor_petal_width)
    print(f"Pearson Correlation Coefficient: {r}")
    return r


def main():
    set_seaborn_styling()
    iris_dataset = eda()
    iris_data = iris_dataset['data']
    iris_target = iris_dataset['target']
    setosa_petal_length = get_setosa_petal_length(iris_data, iris_target)
    versicolor_petal_length = get_versicolor_petal_length(iris_data, iris_target)
    versicolor_petal_width = get_versicolor_petal_width(iris_data, iris_target)
    virginica_petal_length = get_virginica_petal_length(iris_data, iris_target)

    plot_versicolor_petal_length(versicolor_petal_length)
    iris_df = get_all_dataframe(iris_data, iris_target)
    plot_swarm(iris_df)
    plot_initial_ecdf(versicolor_petal_length)
    plot_comparison_of_ecdfs(setosa_petal_length, versicolor_petal_length, virginica_petal_length)

    setosa_mean, versicolor_mean, virginica_mean = compute_means(setosa_petal_length, versicolor_petal_length, virginica_petal_length)
    setosa_percentile, versicolor_percentile, virginica_percentile = compute_percentiles(setosa_petal_length, versicolor_petal_length, virginica_petal_length)
    compare_versicolor_ecdf_to_percentiles(versicolor_petal_length)
    generate_box_and_whisker(iris_df)

    vers_var = compute_explicit_and_calculated_variances(versicolor_petal_length)
    vers_std = compute_explicit_and_calculated_stds(versicolor_petal_length)

    generate_scatter_plot(versicolor_petal_length, versicolor_petal_width)
    vers_covariance_matrix, vers_covariance = calculate_covariance(versicolor_petal_length, versicolor_petal_width)
    pearson_correlation_coefficient = calculate_pearson_r(versicolor_petal_length, versicolor_petal_width)


if __name__ == '__main__':
    main()
