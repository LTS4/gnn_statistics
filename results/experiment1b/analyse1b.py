import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)

path_to_results = None      # Fill with the path to raw results
df_n_features = pd.read_csv('results/experiment1b/' + path_to_results)

# A. Plot the performance of both methods
for i, dataset in enumerate(['cora', 'citeseer', 'pubmed']):
    df_cora = df_n_features[df_n_features['dataset'] == dataset]

    train_proportion = np.unique(df_cora['train_proportion'])

    plt.figure()

    for method in ['GCN', 'SGC']:
        mean = df_cora.groupby(['model_name', 'train_proportion'])['test.accuracy'].mean()[method].to_numpy()
        mean = mean * 100  # Compute in %
        std = df_cora.groupby(['model_name', 'train_proportion'])['test.accuracy'].std()[method].to_numpy()
        std = std * 100
        # To change: don't keep results with too few features

        confidence = 2.021 * std / np.sqrt(40)

        zipped = [(x - y, x + y) for x, y in zip(mean, confidence)]

        remove_first = 0
        mean = mean[remove_first:]
        std = std[remove_first:]
        confidence = confidence[remove_first:]

        plt.plot(train_proportion[remove_first:], mean, label=method)
        plt.fill_between(train_proportion[remove_first:], mean - confidence, mean + confidence, alpha=0.3)

        if method == 'GCN':
            std_gcn = std
            mean_gcn = mean
        else:
            std_sgc = std
            mean_sgc = mean

    if i == 0:
        plt.legend()  # Print legend and y axis only for the first dataset
        plt.ylabel("Accuracy (%)")
    plt.title("Classification performance on " + dataset.upper())

    # B. Plot the relative performance
    diff = mean_gcn - mean_sgc
    conf = 2.021 * np.sqrt(std_gcn ** 2 / 40 + std_sgc ** 2 / 40)
    plt.figure(figsize=(6.4, 4))
    plt.plot(train_proportion[remove_first:], diff)
    plt.fill_between(train_proportion[remove_first:], diff - confidence, diff + confidence, alpha=0.3)
    plt.xlabel("Proportion of observed nodes")
    if i == 0:
        plt.ylabel("Accuracy (%)")
    plt.title("Difference in accuracy between GCN and SGC")

plt.show()
