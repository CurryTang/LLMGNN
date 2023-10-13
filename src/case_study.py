import torch
import os.path as osp
import ipdb 
from helper.args import get_command_line_args
from helper.utils import load_yaml, replace_tensor_values, noise_transition_matrix
from helper.utils import get_one_hop_neighbors, get_two_hop_neighbors_no_multiplication, get_sampled_nodes
from helper.train_utils import graph_consistency
from helper.data import get_dataset
from helper.active import density_query
import numpy as np
import matplotlib.pyplot as plt


def case_study(data_path, datasets = ['cora', 'pubmed']):
    ## 1. What's the propotion of the data that ground truth will appear in topK labels

    ## load pl
    for dataset in datasets:
        data = torch.load(osp.join(data_path, f"{dataset}_fixed_pl.pt"))
        s_data = torch.load(osp.join(data_path, f"{dataset}_fixed_sbert.pt"))
        pseudo_labels = data.x
        pseudo_labels -= 1
        pl_list = pseudo_labels.tolist()
        if dataset == 'cora':
            mapping = {0: 2, 1:3, 2:1, 3:6, 4:5, 5:0, 6:4}
            pseudo_labels = replace_tensor_values(pseudo_labels, mapping)
        # import ipdb; ipdb.set_trace()
        gt = s_data.y 
        top1_pred = pseudo_labels[:, 0]
        full_match = (top1_pred == gt).sum().item()

        # First, we need to make the dimensions of ground_truth match with preds
        ground_truth = gt.view(-1, 1)

        # Now, check if the ground_truth values are in preds
        mask = pseudo_labels.eq(ground_truth)

        # Count how many times ground_truth appears in preds
        count = mask.any(dim=1).sum().item()

        print(f" {dataset} Full match ratio: ", full_match / len(gt))
        print(f" {dataset} Partial match ratio: ", count / len(gt))

        plot_name = f"{dataset}_noise_transition_matrix.png"

        # noise_transition_matrix(top1_pred, gt, plot_name)
        ## density based
        x_embed = data.x
        n_cluster = data.y.max().item() + 1
        train_mask = torch.ones(len(data.y), dtype = torch.bool)
        seed = 0
        device = 'cuda'
        _, density = density_query(x_embed.shape[0], x_embed, n_cluster, train_mask, seed, device)

        sorted_indices = torch.argsort(density, descending=True)

        non_empty_column = (top1_pred[sorted_indices] != -1)

        sorted_pred = top1_pred[sorted_indices][non_empty_column]
        sorted_y = data.y[sorted_indices][non_empty_column]
        # sorted_conf = conf[sorted_indices][non_empty_column]

        num_regions = 20
        samples_per_region = len(sorted_y) // num_regions
        regions = range(num_regions)
        accuracies = []

        for i in regions:
            start_idx = i * samples_per_region
            end_idx = start_idx + samples_per_region
            accuracy = (sorted_pred[start_idx:end_idx] == sorted_y[start_idx:end_idx]).float().mean()
            accuracies.append(accuracy.item())
        
        plt.bar(regions, accuracies, alpha=0.8, color='blue')
        plt.xlabel("Region Index")
        plt.ylabel("Average Accuracy")
        plt.title("Accuracy by Region")
        plt.xticks(regions)
        plt.ylim([0, 1])
        plt.show()
        plt.savefig("{}_active.png".format(dataset))

        # if dataset == 'cora':
        #     _, density = density_query(b, x_embed, n_cluster, train_mask, seed, device):


            






    ## load dataset

    
    ##2. What's the accuracy of only 1 predictions
            # Find where tensor is not -1 (i.e., non-empty), this gives a tensor of the same shape with True where it's not -1
        # non_empty = pseudo_labels.ne(-1)
        # # Count the number of non-empty entries along each row
        # count = non_empty.sum(dim=1)
        # # Find where the count is exactly 1
        # rows = (count == 1).nonzero(as_tuple=True)[0]
        # count = (gt[rows] == top1_pred[rows]).sum().item()
        # total_count = len(rows)
        # print(f"{dataset} only 1 label acc: ", count / total_count)
        # print(f"{dataset} has {total_count} samples with only 1 label")
    
        # ## Consistency based on 1 hop neighbors
        # test_node_idxs, train_node_idxs = get_sampled_nodes(s_data)
        # one_hop_neighbor_dict = get_one_hop_neighbors(s_data, test_node_idxs)
        # two_hop_neighbor_dict = get_two_hop_neighbors_no_multiplication(s_data, test_node_idxs)

        # sorted_nodes = graph_consistency(one_hop_neighbor_dict, pseudo_labels, gt, top1_pred)
        # sorted_nodes = sorted_nodes[::-1]
        # for i in range(5):
        #     this_sorted_nodes = sorted_nodes[:len(sorted_nodes) // 5 * (i + 1)]
        #     this_part_pred = torch.tensor([top1_pred[key] for key in this_sorted_nodes])
        #     this_part_gt = torch.tensor([gt[key] for key in this_sorted_nodes])
        #     acc = (this_part_pred == this_part_gt).sum().item() / len(this_part_gt)
        #     print("This sorted acc:", acc)

        
        



        

if __name__ == '__main__':
    command_line_args = get_command_line_args()
    params_dict = load_yaml(command_line_args.yaml_path)
    data_path = params_dict['DATA_PATH']
    case_study(data_path)
        


