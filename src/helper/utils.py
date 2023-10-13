import yaml
import pickle as pkl
import os
import torch
import requests
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
        except yaml.YAMLError as e:
            print(f"Error while parsing YAML file: {e}")


def pkl_and_write(obj, path):
    directory = os.path.dirname(path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
    return path

def read_and_unpkl(path):
    with open(path, 'rb') as f:
        res = pkl.load(f)
    return res 


def replace_tensor_values(tensor, mapping):
    # Create an empty tensor with the same shape as the original tensor
    new_tensor = torch.zeros_like(tensor)
    new_tensor -= 1

    # Loop over the mapping and replace the values
    for k, v in mapping.items():
        mask = tensor == k  # create a mask where the tensor value equals the current key
        new_tensor[mask] = v 
    return new_tensor


def neighbors(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()

def get_one_hop_neighbors(data_obj, sampled_test_node_idxs, sample_num = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(data_obj.edge_index, center_node_idx)
    return neighbor_dict

def get_two_hop_neighbors_no_multiplication(data_obj, sampled_test_node_idxs, sample_num = -1):
    neighbor_dict = {}
    # for center_node_idx in sampled_test_node_idxs:
    one_hop_neighbor_dict = get_one_hop_neighbors(data_obj, sampled_test_node_idxs)
    for key, value in one_hop_neighbor_dict.items():
        this_key_neigh = []
        second_hop_neighbor_dict = get_one_hop_neighbors(data_obj, torch.IntTensor(value))
        second_hop_neighbors = set(itertools.chain.from_iterable(second_hop_neighbor_dict.values()))
        second_hop_neighbors.discard(key)
        neighbor_dict[key] = sorted(list(second_hop_neighbors))
    return neighbor_dict


def get_sampled_nodes(data_obj, sample_num = -1):
    train_mask = data_obj.train_masks[0]
    # val_mask = data_obj.val_masks[0]
    test_mask = data_obj.test_masks[0]
    all_idxs = torch.arange(data_obj.x.shape[0])
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    # val_node_idxs = all_idxs[val_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    return sampled_test_node_idxs, train_node_idxs


def query_arxiv_classify_api(title, abstract, url = "http://export.arxiv.org/api/classify"):
    text = title + abstract
    data = {
        "text": text
    }
    r = requests.post(url, data = data)
    return r



def plot_multiple_curve(methods_acc, method_name, output_path):
    markers = ["*-r", "v-b", "o-c", "^-m", "<-y", ">-k"]
    method1_acc = methods_acc[0]
    epochs = range(1, len(method1_acc) + 1)
    plt.figure(figsize=(10,6))

    # Plotting each method's accuracy over epochs
    for i, acc in enumerate(methods_acc):
        plt.plot(epochs, acc, markers[i], label=method_name[i])

    # Adding labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(output_path)


def noise_transition_matrix(pred, gt, output_path, x_axis_labels = None, y_axis_labels = None, cbar = True):
    plt.figure(figsize=(12, 12))
    pred_list = pred.tolist()
    gt_list = gt.tolist()
    num_class = gt.max().item() + 1
    transition_matrix = np.zeros((num_class, num_class))
    num_of_occurence = np.bincount(gt_list)
    for x, y in zip(pred_list, gt_list):
        transition_matrix[y][x] += 1
    # pal = sns.color_palette("crest", as_cmap=True)
    cmap_pal = sns.color_palette("crest", as_cmap=True)
    # sns.set_palette(pal)
    transition_matrix /= num_of_occurence[:, np.newaxis]
    if x_axis_labels is None or y_axis_labels is None:
        ax = sns.heatmap(transition_matrix, vmin=0, vmax=1,  annot=True, fmt=".2f", norm = LogNorm(), cmap = cmap_pal, cbar = cbar, annot_kws={"size": 20}, square = True, cbar_kws={'shrink': 0.5})
    else:
        ax = sns.heatmap(transition_matrix, vmin=0, vmax=1, annot=True, fmt=".2f", norm = LogNorm(), xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap = cmap_pal, cbar = cbar, annot_kws={"size": 20}, square = True, cbar_kws={'shrink': 0.5})
    plt.tight_layout()
    # Adjust x-axis label properties
    plt.xticks(fontsize=40, fontweight='bold')

    # Adjust y-axis label properties
    plt.yticks(fontsize=40, fontweight='bold')
    # cbar = ax.collections[0].colorbar
    # # And an example of setting custom ticks:
    # cbar.set_ticks([0, 1])
    plt.savefig(output_path)
    plt.clf()




def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data