from openail.utils import load_partial_openai_result
import ipdb
from llm import Experiment, few_shot_all_query
from helper.data import get_dataset
from helper.active import train_lr, inference_lr, density_query
import torch
import matplotlib.pyplot as plt
from models.nn import LinearRegression
from openail.utils import load_mapping
import numpy as np
from helper.train_utils import seed_everything

plt.figure(figsize=(12, 6))

data_path = "../../ogb/preprocessed_data/new"
datasets = ['cora', 'citeseer', 'wikics', 'pubmed']
prompt_key = "consistency"
seeds = [5]
for dataset in datasets:
    # exp = Experiment(dataset, data_path)
    random_data = get_dataset(seeds, dataset, 'active', 'sbert', data_path, None, random_noise = 0, no_val = 1, budget = 20, strategy = 'random', num_centers = 0, compensation = 0, save_data = 0, llm_strategy = 'none', max_part = 0, oracle_acc = 1, reliability_list = None, total_budget = 140, second_filter = None)
    exp = Experiment(random_data, None, None)
    total_idxs = torch.arange(len(random_data.y))
    idxs = total_idxs[random_data.train_masks[0]]
    selected_features = random_data.x[idxs]
    cache = load_partial_openai_result(data_path, dataset, prompt_key)
    few_shot_result = cache[prompt_key]
    selected_result = [few_shot_result[idx] for idx in idxs]
    saved_results = torch.load("../../ogb/preprocessed_data/new/active/{}^cache^consistency.pt".format(dataset))
    pred = saved_results['pred']
    conf = saved_results['conf']

    non_empty_column = (pred != -1)
    non_empty_idxs = total_idxs[non_empty_column]
    seed_everything(5)
    non_empty_column = torch.randperm(non_empty_idxs.shape[0])[:1000]
    # new_idxs = torch.randperm(total_idxs[non_empty_column].shape[0])
    # select_idxs = total_idxs[non_empty_column][new_idxs][:50]
    # random_confidence = conf[select_idxs]
    # random_x = random_data.x[select_idxs]
    # x_embed = random_data.x
    # lr_model = LinearRegression(x_embed.shape[1])
    # train_lr(lr_model, random_x, random_confidence, 100)
    # lr_pred = inference_lr(lr_model, x_embed).reshape(-1)

    # new_mask = torch.ones(len(random_data.y), dtype = torch.bool)
    # new_mask[select_idxs] = 0
    # max_conf_id = torch.argsort(lr_pred[new_mask], descending=True)[:140]

    raw_texts = random_data.raw_texts
    # data.label_names = [full_mapping[dataname][x] for x in data.label_names]
    # data.label_names = [x.lower() for x in data.label_names]



    # print()



    x_embed = random_data.x
    n_cluster = random_data.y.max().item() + 1
    train_mask = torch.ones(len(random_data.y), dtype = torch.bool)
    seed = 0
    device = 'cuda'
    _, density = density_query(x_embed.shape[0], x_embed, n_cluster, train_mask, seed, device)

    sorted_indices = torch.argsort(density, descending=True)

    non_empty_column = (pred[sorted_indices] != -1)

    sorted_pred = pred[sorted_indices][non_empty_column][:1000]
    sorted_y = random_data.y[sorted_indices][non_empty_column][:1000]
    sorted_conf = conf[sorted_indices][non_empty_column][:1000]

    num_regions = 10
    samples_per_region = len(sorted_conf) // num_regions
    regions = range(num_regions)
    confs = []
    accuracies = []
    t_acc = []

    for i in regions:
        start_idx = i * samples_per_region
        end_idx = start_idx + samples_per_region
        conf = (sorted_conf[start_idx:end_idx]).mean()
        accuracy = (sorted_pred[start_idx:end_idx] == sorted_y[start_idx:end_idx]).float().mean()
        ta = (sorted_pred[:end_idx] == sorted_y[:end_idx]).float().mean()
        accuracies.append(accuracy.item())
        t_acc.append(ta.item())
        confs.append(conf.item())
    
    # plt.bar(regions, accuracies, alpha=0.8, color='blue')
    plt.bar(regions, accuracies, alpha=0.8, color='green')
    plt.plot(regions, t_acc, color = 'blue', linewidth = 3, marker = 'o', markersize = 12)
    plt.xlabel("Region Index", fontsize = 25, fontweight = 'bold')
    plt.ylabel("Average Accuracy", fontsize = 25, fontweight = 'bold')
    # plt.title("Accuracy by Region")
    plt.xticks(regions, fontsize = 20, fontweight = 'bold')
    plt.yticks(fontsize = 20, fontweight = 'bold')
    plt.ylim([0.6, 1])
    # plt.show()
    plt.tight_layout()
    # plt.savefig("{}_active_conf.png".format(dataset))
    # plt.savefig("{}_active_conf.pdf".format(dataset))
    plt.savefig("{}_active.png".format(dataset))
    plt.savefig("{}_active.pdf".format(dataset))

    plt.clf()

    # import ipdb; ipdb.set_trace()
    