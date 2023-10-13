import torch 
import matplotlib.pyplot as plt
from helper.train_utils import seed_everything

datasets = ['cora', 'citeseer', 'pubmed', 'wikics']
prompts = ['few_shot', 'consistency_0_vanilla', 'topk', 'few_shot_all', 'zero_shot', 'consistency']
ks = [50, 100, 150, 200, 250, 300]

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['xtick.labelsize'] = 25   # Set x-tick font size
plt.rcParams['ytick.labelsize'] = 25   # Set y-tick font size
plt.rcParams['font.weight'] = 'bold'   # Set font weight to bold
plt.rcParams['font.size'] = 25         # Set font size to 16
plt.rcParams['axes.labelweight'] = 'bold' # Set font weight of axis labels to bold
plt.rcParams['axes.labelsize'] = 25       # Set font size of axis labels to 16
plt.rcParams['font.family'] = 'sans-serif'  # Set font family to serif

styles = ['-', '--', '-.', ':', '-']
markers = ['.', ',', 'o', 'v', '^', '<', '>']


def plot_single_lines(x, ys, labels, title, xlabel, ylabel, save_path, save_path2, show_legend = True):
    plt.figure(figsize=(14, 7))
    for i, (y, label) in enumerate(zip(ys, labels)):
        if label == 'consistency_0_vanilla':
            slabel = 'Most Voting'
        if label == 'few_shot_all':
            slabel = 'Hybrid(One-Shot)'
        if label == 'few_shot':
            slabel = 'One-Shot'
        if label == 'zero_shot':
            slabel = 'Zero-Shot'
        if label == 'consistency':
            slabel = 'Hybrid(Zero-Shot)'
        if label == 'topk':
            slabel = 'Top-K'
        plt.plot(x, y, label=slabel, linewidth=2, marker=markers[i], markersize=10)
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_legend:
        plt.legend(fontsize = 17)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path2)
    plt.show()

seed_everything(2)

for d in datasets:
    alls = []
    for p in prompts:
        results = []
        gt = torch.load("../../ogb/preprocessed_data/new/{}_fixed_sbert.pt".format(d), map_location='cpu').y
        filee = torch.load("../../ogb/preprocessed_data/new/active/{}^cache^{}.pt".format(d, p), map_location='cpu')
        pred = filee['pred']
        conf = filee['conf']
        total_idxs = torch.arange(pred.shape[0])
        non_zero_mask = (pred != -1)
        non_zero_idx = total_idxs[non_zero_mask]
        # pred = pred[non_zero_idx]
        # conf = conf[non_zero_idx]
        indices = non_zero_idx[torch.randperm(non_zero_idx.size(0))][:300]
        non_zero_pred = pred[indices]
        non_zero_conf = conf[indices]
        non_zero_y = gt[indices]
        idxs = torch.argsort(non_zero_conf, descending=True)
        for k in ks:    
            non_zero_pred_this = non_zero_pred[idxs][:k]
            # non_zero_conf_this = non_zero_conf[idxs][:k]
            non_zero_y_this = non_zero_y[idxs][:k]
            results.append((non_zero_pred_this == non_zero_y_this).float().mean().item())
            
        alls.append(results)
    if d == 'wikics':
        showgg = True 
    else:
        showgg = False
    plot_single_lines(ks, alls, prompts, d, 'k', 'Accuracy', '{}.png'.format(d), '{}.pdf'.format(d), showgg)


