import torch
import os.path as osp
import ipdb 
from helper.args import get_command_line_args
from helper.utils import load_yaml, read_and_unpkl
from helper.utils import plot_multiple_curve
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


plt.style.use('seaborn')
plt.rcParams['xtick.labelsize'] = 20   # Set x-tick font size
plt.rcParams['ytick.labelsize'] = 20   # Set y-tick font size
plt.rcParams['font.weight'] = 'bold'   # Set font weight to bold
plt.rcParams['font.size'] = 20         # Set font size to 16
plt.rcParams['axes.labelweight'] = 'bold' # Set font weight of axis labels to bold
plt.rcParams['axes.labelsize'] = 20       # Set font size of axis labels to 16
plt.rcParams['font.family'] = 'sans-serif'  # Set font family to serif
import matplotlib.pyplot as plt

# Data from the table
x_axis = [70, 140, 280, 560, 1120, 2240]
Random = [66.15, 70.48, 70.63, 72.07, 72.67, 74.79]
PS_Random = [65.16, 69.83, 70.45, 71.64, 72.96, 74.02]
FeatProp = [69.33, 72.59, 70.72, 72.67, 73.74, 76.71]
PS_Featprop = [69.72, 75.54, 72.22, 73.1, 73.84, 76.7]
DA_Degree = [70.55, 74.64, 70.26, 70.76, 69.96, 70.94]

# Plotting the data
plt.plot(x_axis, Random, marker='o', label='Random')
plt.plot(x_axis, PS_Random, marker='.', label='PS-Random')
plt.plot(x_axis, FeatProp, marker='^', label='FeatProp')
plt.plot(x_axis, PS_Featprop, marker='>', label='PS-Featprop')
plt.plot(x_axis, DA_Degree, marker='*', label='DA-Degree')

# Setting labels, title, legend, and grid
plt.xlabel("Budget", fontweight='bold')
plt.ylabel("Accuracy", fontweight='bold')
# plt.title("Line Plot from Table Data")
plt.legend(fontsize = 15)
plt.xticks([70, 280, 1120, 2240])
# plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig('line_plot.png')
plt.savefig('line_plot.pdf')

plt.show()




# def plot_accuracy(accuracies1, accuracies2, accuracies3, accuracies4, accuracies5, accuracies6, accuracies7, accuracies8, color = ['blue'], title = 'test'):
#     """
#     Plot the average accuracy across epochs with variance among seeds.

#     Parameters:
#     - accuracies: List of lists containing accuracy values for each seed across epochs.
#     """
#     # Set the figure size
#     plt.figure(figsize=(8, 6))  # 12 inches by 8 inches

#     # Set global text sizes
#     plt.rcParams['font.size'] = 16         # Default font size
#     plt.rcParams['axes.titlesize'] = 16    # Axes title size
#     plt.rcParams['axes.labelsize'] = 16    # X and Y axes label size
#     plt.rcParams['xtick.labelsize'] = 16   # X-tick label size
#     plt.rcParams['ytick.labelsize'] = 16   # Y-tick label size
#     plt.rcParams['legend.fontsize'] = 10   # Legend font size
#     plt.rcParams['font.family'] = 'sans-serif' # Figure title size
#     color = ['blue', 'blue', 'red', 'red', 'green', 'green', 'yellow', 'yellow']
    
#     line_style = ['-', '--', '-', '--', '-', '--', '-', '--']

#     names = ['Test acc LLM', 'Train acc LLM', 'Test acc gt all', 'Train acc gt all', 'Test acc syn', 'Train acc syn', 'Test acc filter', 'Train acc filter']

#     for i, accuracies in enumerate([accuracies1, accuracies2, accuracies3, accuracies4, accuracies5, accuracies6, accuracies7, accuracies8]):

#         c = color[i]

#         ls = line_style[i]

#         ll = names[i]

#         # Convert accuracies to a numpy array for easier calculations
#         accuracies = np.array(accuracies)

#         # Calculate the mean and standard deviation across seeds for each epoch
#         mean_accuracies = np.mean(accuracies, axis=0)
#         std_accuracies = np.std(accuracies, axis=0)

#         # Define the epochs
#         epochs = np.arange(1, 151)
#         # Plot the mean accuracy curve
#         plt.plot(epochs, mean_accuracies, label=ll, color=c, linestyle = ls)

#         # Fill between for variance (mean +/- std)
#         plt.fill_between(epochs, mean_accuracies - std_accuracies, mean_accuracies + std_accuracies, color=c, alpha=0.2)

#     # Set the x-axis and y-axis limits and labels
#     plt.xticks(np.arange(0, 151, 25))
#     plt.xlim(0, 150)
#     plt.ylim(np.min(accuracies4), 1.01)
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')

#     # Display the legend and the plot
#     plt.legend(loc='best', ncol = 2)
#     # plt.title(title)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(title + '.pdf')
#     plt.savefig(title + '.png')
#     # plt.show()

    



# def case_study():
#     ## GCN 
#     pubmed_gcn_noisy = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_60_random_consistency.pkl")

#     pubmed_gcn_noisy_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_60_random_consistency_train_accs.pkl")

#     pubmed_gcn_gt = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_60_random_few_shot_all_gt.pkl")

#     pubmed_gcn_gt_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_60_random_few_shot_all_train_accs_gt.pkl")

#     pubmed_gcn_filter = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_60_random_consistency_filtered.pkl")

#     pubmed_gcn_filter_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_60_random_consistency_train_accs_filtered.pkl")

#     pubmed_gcn_syn = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_1.0_20_60_random_consistency.pkl")

#     pubmed_gcn_syn_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_1.0_20_60_random_consistency_train_accs.pkl")
    
#     plot_accuracy(pubmed_gcn_noisy, pubmed_gcn_noisy_train, pubmed_gcn_gt, pubmed_gcn_gt_train, pubmed_gcn_syn, pubmed_gcn_syn_train, pubmed_gcn_filter, pubmed_gcn_filter_train,  color='blue', title='pubmed (budget 60)')

#     plt.clf()

#     pubmed_gcn_noisy = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_120_random_consistency.pkl")

#     pubmed_gcn_noisy_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_120_random_consistency_train_accs.pkl")

#     pubmed_gcn_gt = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_120_random_few_shot_all_gt.pkl")

#     pubmed_gcn_gt_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_120_random_few_shot_all_train_accs_gt.pkl")

#     pubmed_gcn_filter = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_0.0_20_120_random_consistency_filtered.pkl")

#     pubmed_gcn_filter_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_0.0_20_120_random_consistency_train_accs_filtered.pkl")

#     pubmed_gcn_syn = read_and_unpkl("../../annotation/src/scripts/debug/GCN_active_pubmed_sbert_1.0_20_120_random_consistency.pkl")

#     pubmed_gcn_syn_train = read_and_unpkl("../../annotation/src/scripts/debug_train/GCN_active_pubmed_sbert_1.0_20_120_random_consistency_train_accs.pkl")
    
#     plot_accuracy(pubmed_gcn_noisy, pubmed_gcn_noisy_train, pubmed_gcn_gt, pubmed_gcn_gt_train, pubmed_gcn_syn, pubmed_gcn_syn_train, pubmed_gcn_filter, pubmed_gcn_filter_train,  color='blue', title='pubmed (budget 120)')
    
#     # plot_accuracy(pubmed_gcn_noisy, pubmed_gcn_noisy_train, pubmed_gcn_gt, pubmed_gcn_gt_train, pubmed_gcn_syn, pubmed_gcn_syn_train,  color='blue', title='pubmed (budget 120)')

    


# case_study()
