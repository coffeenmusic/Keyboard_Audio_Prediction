import numpy as np
import collections
import matplotlib.pyplot as plt

width = 0.5  # Bar width
figsize = (15, 4)

def plot_key_hist(most_common):
    if len(most_common) > 30:
        most_common = most_common[:30]

    hist_labels, hist_values = zip(*most_common)  # Show only a subset of all keys
    indexes = np.arange(len(hist_labels))
    plt.figure(figsize=figsize)
    plt.bar(indexes, hist_values, width)
    plt.xticks(indexes + width * 0.5, hist_labels, rotation='vertical')
    plt.show()

def plot_trunc_key_cnt(key_cnt, trunc_cnt):
    if len(key_cnt) > 30:
        key_cnt = key_cnt[:30]

    hist_labels, hist_values = zip(*key_cnt)  # Show only a subset of all keys
    hist_idx = np.arange(len(hist_labels))
    plt.figure(figsize=figsize)
    p1 = plt.bar(hist_idx, hist_values, width)

    trunc_labels, trunc_values = zip(*trunc_cnt)  # Show only a subset of all keys
    trunc_idx = np.arange(len(trunc_labels))
    p2 = plt.bar(trunc_idx, trunc_values, width, color='#d62728')

    plt.ylabel('Key Counts')
    plt.title('Truncated Key Counts')
    plt.xticks(hist_idx + width * 0.5, hist_labels, rotation='vertical')
    plt.legend((p1[0], p2[0]), ('Counts', 'Truncated'))
    plt.show()

# Maximize DataSet Size By Finding Optimal Threshold
def get_optimized_threshold(key_count):
    max_data_size = 0
    max_class_cnt = 0
    threshold_max = 0
    for _, v in key_count.most_common():
        threshold = v - 1
        min_thresh_cnts = [count for key, count in zip(key_count.keys(), key_count.values()) if count > threshold]
        class_cnt = len(min_thresh_cnts)
        data_size = min(min_thresh_cnts)*len(min_thresh_cnts)
        if data_size > max_data_size:
            max_data_size = data_size
            max_class_cnt = class_cnt
            threshold_max = threshold
    return max_data_size, max_class_cnt, threshold_max