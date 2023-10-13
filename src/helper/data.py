import torch 
import os.path as osp
from helper.train_utils import seed_everything
from helper.active import *
import numpy as np 
from torch_geometric.utils import index_to_mask
import random
from helper.utils import get_one_hop_neighbors, get_two_hop_neighbors_no_multiplication, get_sampled_nodes
from helper.train_utils import graph_consistency
from torch_geometric.typing import SparseTensor
from helper.partition import GraphPartition
import networkx as nx
import ipdb
import os
import torch_geometric.transforms as T
from tqdm import tqdm
# from ..llm import query


class LabelPerClassSplit:
    def __init__(
            self,
            num_labels_per_class: int = 20,
            num_valid: int = 500,
            num_test: int = -1,
            inside_old_mask: bool = False
    ):
        self.num_labels_per_class = num_labels_per_class
        self.num_valid = num_valid
        self.num_test = num_test
        self.inside_old_mask = inside_old_mask

    def __call__(self, data, total_num, split_id = 0):
        new_train_mask = torch.zeros(total_num, dtype=torch.bool)
        new_val_mask = torch.zeros(total_num, dtype=torch.bool)
        new_test_mask = torch.zeros(total_num, dtype=torch.bool)

        if self.inside_old_mask:
            old_train_mask = data.train_masks[split_id]
            old_val_mask = data.val_masks[split_id]
            old_test_mask = data.test_masks[split_id]
            perm = torch.randperm(total_num)
            train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int)

            for i in range(perm.numel()):
                label = data.y[perm[i]]
                if train_cnt[label] < self.num_labels_per_class and old_train_mask[perm[i]].item():
                    train_cnt[label] += 1
                    new_train_mask[perm[i]] = 1
                elif new_val_mask.sum() < self.num_valid and old_val_mask[perm[i]].item():
                    new_val_mask[perm[i]] = 1
                else:
                    if self.num_test != -1:
                        if new_test_mask.sum() < self.num_test and old_test_mask[perm[i]].item():
                            new_test_mask[perm[i]] = 1
                    else:
                        new_test_mask[perm[i]] = 1

            
            return new_train_mask, new_val_mask, new_test_mask
        else:
            perm = torch.randperm(total_num)
            train_cnt = np.zeros(data.y.max().item() + 1, dtype=np.int32)

            for i in range(perm.numel()):
                label = data.y[perm[i]]
                if train_cnt[label] < self.num_labels_per_class:
                    train_cnt[label] += 1
                    new_train_mask[perm[i]] = 1
                elif new_val_mask.sum() < self.num_valid:
                    new_val_mask[perm[i]] = 1
                else:
                    if self.num_test != -1:
                        if new_test_mask.sum() < self.num_test:
                            new_test_mask[perm[i]] = 1
                        else:
                            new_test_mask[perm[i]] = 1

            
            return new_train_mask, new_val_mask, new_test_mask

def generate_random_mask(total_node_number, train_num, val_num, test_num = -1, seed = 0):
    seed_everything(seed)
    random_index = torch.randperm(total_node_number)
    train_index = random_index[:train_num]
    val_index = random_index[train_num:train_num + val_num]
    if test_num == -1:
        test_index = random_index[train_num + val_num:]
    else:
        test_index = random_index[train_num + val_num: train_num + val_num + test_num]
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)

def get_different_num(num, k):
    possible_nums = list(set(range(k)) - {num})
    return random.choice(possible_nums)

def inject_random_noise(data_obj, noise):
    t_size = data_obj.x.shape[0]
    idx_array = torch.arange(t_size)
    k = data_obj.y.max().item() + 1
    ys = []
    for i in range(len(data_obj.train_masks)):
        train_mask = data_obj.train_masks[i]
        val_mask = data_obj.val_masks[i]
        selected_idxs = torch.randperm(t_size)[:int(t_size * noise) + 1]
        this_y = data_obj.y.clone()
        new_y = torch.LongTensor([get_different_num(num.item(), k) if i in selected_idxs else num.item() for i, num in enumerate(this_y)])
        this_y[train_mask] = new_y[train_mask]
        this_y[val_mask] = new_y[val_mask]
        ys.append(this_y)
        # new_train_masks.append(train_mask)
    data_obj.ys = ys
    return data_obj

def inject_random_noise_y_level(orig_y, noise):
    t_size = len(orig_y)
    idx_array = torch.arange(t_size)
    k = orig_y.max().item() + 1
    dirty_y = orig_y.clone()
    selected_idxs = torch.randperm(t_size)[:int(t_size * noise) + 1]
    new_y = torch.LongTensor([get_different_num(num.item(), k) if i in selected_idxs else num.item() for i, num in enumerate(dirty_y)])
    # ys.append(this_y)
        # new_train_masks.append(train_mask)
    return new_y




def generate_pl_mask(data, no_val = 0, total_budget = 140, seed = 0):
    seed_everything(seed)
    num_classes = data.y.max().item() + 1
    total_num = data.x.shape[0]
    total_label_num = total_budget
    if no_val:
        train_num = total_label_num
        val_num = 0
    else:
        train_num = total_label_num * 3 // 4
        val_num = total_label_num - train_num
    test_num = int(total_num * 0.2)
    t_mask, val_mask, test_mask = generate_random_mask(data.x.shape[0], train_num, val_num, test_num, seed = seed)
    return t_mask, val_mask, test_mask





def get_active_dataset(seeds, orig_data, dataset, strategy, llm_strategy, data_path, second_filter):
    ys = []
    confs = []
    test_masks = []
    train_masks = []
    val_masks = []
    active_path = osp.join(data_path, 'active')
    gt = orig_data.y
    label_quality = []
    valid_nums = []
    for s in seeds:
        data_file = torch.load(osp.join(active_path, f"{dataset}^result^{strategy}^{llm_strategy}^{s}.pt"), map_location='cpu')
        pred = data_file['pred']
        conf = data_file['conf']
        test_mask = data_file['test_mask']
        test_masks.append(test_mask)
        train_mask = pred != -1
        train_masks.append(train_mask)
        val_mask = torch.zeros_like(train_mask)
        val_masks.append(val_mask)
        y_copy = torch.tensor(gt)
        y_copy[train_mask] = pred[train_mask]
        ys.append(y_copy)
        if second_filter == 'weight':
            _, conf = active_llm_query(train_mask.sum(), orig_data.edge_index, orig_data.x, conf, s, train_mask, orig_data)
            # train_mask = graph_consistency(orig_data, train_mask, s)
            # train_masks[-1] = train_mask
        confs.append(conf)
        valid_nums.append(train_mask.sum().item())
        label_quality.append((y_copy[train_mask] == gt[train_mask]).sum().item() / train_mask.sum().item())
    orig_data.ys = ys
    orig_data.confs = confs
    orig_data.test_masks = test_masks
    orig_data.train_masks = train_masks
    orig_data.val_masks = val_masks
    mean_acc = np.mean(label_quality) * 100
    std_acc = np.std(label_quality) * 100
    print(f"Label quality: {mean_acc:.2f} ± {std_acc:.2f} ")
    print(f"Valid num: {np.mean(valid_nums)} ± {np.std(valid_nums)}")
    return orig_data



def select_random_indices(tensor, portion):
    """
    Randomly select indices from a tensor based on a given portion.

    Parameters:
    - tensor: The input tensor.
    - portion: The portion of indices to select (between 0 and 1).

    Returns:
    - selected_indices: A tensor containing the randomly selected indices.
    """
    total_indices = torch.arange(tensor.size(0))
    num_to_select = int(portion * tensor.size(0))
    selected_indices = torch.randperm(tensor.size(0))[:num_to_select]
    return tensor[selected_indices]


def run_active_learning(budget, strategy, filter_strategy, dataset, oracle_acc = 1, seed_num = 3, alpha = 0.1, beta = 0.1, gamma = 0.1):
    os.system("python3 ../../../annotation/src/llm.py --total_budget {} --split active --main_seed_num {} --oracle {} --strategy {} --dataset {} --filter_strategy {} --no_val 1 --train_stage 0 --alpha {} --beta {} --gamma {}".format(budget, seed_num, oracle_acc, strategy, dataset, filter_strategy, alpha, beta, gamma))



def get_dataset(seeds, dataset, split, data_format, data_path, logit_path, random_noise = 0, no_val = 1, budget = 20, strategy = 'random', num_centers = 0, compensation = 0, save_data = 0, llm_strategy = 'none', max_part = 0, oracle_acc = 1, reliability_list = None, total_budget = -1, second_filter = 'none', train_stage = True, post_pro = False, filter_all_wrong_labels = False, alpha = 0.1, beta = 0.1, gamma = 0.1, ratio = 0.3, fixed_data = True):
    seed_num = len(seeds)
    if 'pl' in split or split == 'active' or split == 'low' or split == 'active_train':
        if dataset == 'arxiv' or dataset == 'products' or dataset == 'wikics' or dataset == '20newsgroup':
            data = torch.load(osp.join(data_path, f"{dataset}_fixed_{data_format}.pt"), map_location='cpu')
        elif dataset == 'tolokers':
            data = torch.load(osp.join(data_path, "tolokers_fixed.pt"), map_location='cpu')
        else:
            data = torch.load(osp.join(data_path, f"{dataset}_random_{data_format}.pt"), map_location='cpu')
    else:
        if dataset == 'arxiv' or dataset == 'products':
            data = torch.load(osp.join(data_path, f"{dataset}_fixed_{data_format}.pt"), map_location='cpu')
        else:
            data = torch.load(osp.join(data_path, f"{dataset}_{split}_{data_format}.pt"), map_location='cpu')
    if fixed_data and dataset == 'citeseer':
        new_c_data = torch.load(osp.join(data_path, "citeseer2_fixed_sbert.pt"))
        data.y = new_c_data.y

    print("Load raw files OK!")
    if dataset == 'products':
        data.y = data.y.squeeze()
    if ('pl' in split or 'active' in split)  and 'noise' not in split and llm_strategy != 'none':
        if train_stage:
            if dataset == 'arxiv' or dataset == 'products':
                run_active_learning(total_budget, strategy, llm_strategy, dataset, oracle_acc, seed_num = 1, alpha = alpha, beta = beta, gamma = gamma)
            else:
                run_active_learning(total_budget, strategy, llm_strategy, dataset, oracle_acc, seed_num = seed_num, alpha = alpha, beta = beta, gamma = gamma)
            print("Annotation done!")
        pl_data_path = osp.join(data_path, "active", f"{dataset}^cache^{llm_strategy}.pt")
        if not osp.exists(pl_data_path):
            pl_data = None
            pseudo_labels = None
            conf = None
        else:
            pl_data = torch.load(pl_data_path, map_location='cpu')
            pseudo_labels = pl_data['pred']
            conf = pl_data['conf']
            # reliability_list.append(conf)
        # if dataset == 'arxiv':
        #     pl_data = torch.load(osp.join(data_path, f"{dataset}_fixed_pl.pt"), map_location = 'cpu')
        # else:
        #     if not osp.exists(osp.join(data_path, f"{dataset}_random_pl.pt")): pl_data = None
        #     else:
        #         pl_data = torch.load(osp.join(data_path, f"{dataset}_random_pl.pt"), map_location = 'cpu')   
        # if pl_data is not None:
        #     pseudo_labels = pl_data.x[:, 0][:]
        #     pseudo_labels -= 1
        #     pl_list = pseudo_labels.tolist()
        #     ## TAPE use a different label index with us, so we have to make a transform here
        #     if dataset == 'cora':
        #         mapping = {0: 2, 1:3, 2:1, 3:6, 4:5, 5:0, 6:4}
        #         pl_list = [mapping[i] for i in pl_list]
        #         pseudo_labels = torch.tensor(pl_list)
        # else:
        #     length = data.x.shape[0]
        #     pseudo_labels = torch.tensor([-1 for _ in range(length)])
    else:
        pl_data = None
        pseudo_labels = None
        conf = None
    
    if split == 'active_train':
        ## load confidence and pred from pt file
        data = get_active_dataset(seeds, data, dataset, strategy, llm_strategy, data_path, second_filter)
        for conf in data.confs:
            reliability_list.append(conf)
        return data


    print("Annotation complete!")


    if 'pl' not in split and 'active' not in split and split != 'low' and strategy == 'no':
        if dataset == "products" or dataset == "arxiv":
            data.train_masks = [data.train_masks[0] for _ in range(seed_num)]
            data.val_masks = [data.val_masks[0] for _ in range(seed_num)]
            data.test_masks = [data.test_masks[0] for _ in range(seed_num)]
        else:
            data.train_masks = [data.train_masks[i] for i in range(seed_num)]
            data.val_masks = [data.val_masks[i] for i in range(seed_num)]
            data.test_masks = [data.test_masks[i] for i in range(seed_num)]
        return data
    new_train_masks = []
    new_val_masks = []
    new_test_masks = []
    ys = []
    num_classes = data.y.max().item() + 1
    real_budget = num_classes * budget if total_budget == -1 else total_budget
    if len(data.y.shape) != 1:
        data.y = data.y.reshape(-1)
    # generate new masks here
    for s in seeds:
        seed_everything(s)
        if split == 'fixed':
            ## 20 per class
            fixed_split = LabelPerClassSplit(num_labels_per_class=20, num_valid = 500, num_test=1000)
            t_mask, val_mask, te_mask = fixed_split(data, data.x.shape[0])
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        elif split == 'low' and (dataset == 'arxiv' or dataset == 'products'):
            low_split = LabelPerClassSplit(num_labels_per_class=budget, num_valid = 0, num_test=0)
            t_mask, _, _ = fixed_split(data, data.x.shape[0])
            val_mask = data.val_masks[0]
            test_mask = data.test_masks[0]
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        elif split == 'pl_random':
            t_mask, val_mask, test_mask = generate_pl_mask(data, no_val, real_budget)
            y_copy = torch.tensor(data.y)
            y_copy[t_mask] = pseudo_labels[t_mask]
            y_copy[val_mask] = pseudo_labels[val_mask]
            ys.append(y_copy)
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(test_mask)
        elif split == 'pl_noise_random':
            t_mask, val_mask, test_mask = generate_pl_mask(data, no_val, real_budget)            
            new_train_masks.append(t_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(test_mask)
        elif split == 'active':
            num_classes = data.y.max().item() + 1
            _, val_mask, test_mask = generate_pl_mask(data, no_val, real_budget)
            # import ipdb; ipdb.set_trace()
            if pseudo_labels is not None:
                pl = pseudo_labels
                if dataset == 'arxiv' or dataset == 'products':
                    train_mask = active_generate_mask(test_mask, data.x, seeds[0], data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset, alpha, beta, gamma)
                else:
                    train_mask = active_generate_mask(test_mask, data.x, s, data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset, alpha, beta, gamma)
                ## second filter
                # import ipdb; ipdb.set_trace()
                pl_data = torch.load(pl_data_path, map_location='cpu')
                conf = pl_data['conf']
                if second_filter != 'none':
                    train_mask = post_process(train_mask, data, conf, pseudo_labels, ratio, strategy=second_filter)
                y_copy = -1 * torch.ones(data.y.shape[0], dtype=torch.long)
                y_copy[train_mask] = pseudo_labels[train_mask]
                y_copy[val_mask] = pseudo_labels[val_mask]
                ys.append(y_copy)
            else:
                pl = None
                if dataset == 'arxiv' or dataset == 'products':
                    train_mask = active_generate_mask(test_mask, data.x, seeds[0], data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset)
                else:
                    train_mask = active_generate_mask(test_mask, data.x, s, data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset)
                y_copy = -1 * torch.ones(data.y.shape[0])
                ys.append(y_copy)
            test_mask = ~train_mask
            if filter_all_wrong_labels and train_stage == True:
                wrong_mask = (y_copy != data.y)
                train_mask = train_mask & ~wrong_mask
            new_train_masks.append(train_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(test_mask)
        elif split == 'active_train':
            num_classes = data.y.max().item() + 1
            _, val_mask, test_mask = generate_pl_mask(data, no_val, real_budget)
            pl = pseudo_labels
            train_mask = active_generate_mask(test_mask, data.x, s, data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset)
            y_copy = -1 * torch.ones(data.y.shape[0], dtype=torch.long)
            y_copy[train_mask] = pseudo_labels[train_mask]
            y_copy[val_mask] = pseudo_labels[val_mask]
            ys.append(y_copy)
            data.conf = conf
            non_empty = (y_copy != -1)
            assert (train_mask != non_empty).sum() == 0
            new_train_masks.append(train_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(test_mask)
        else:
            num_classes = data.y.max().item() + 1
            total_num = data.x.shape[0]
            train_num = int(0.6 * total_num)
            if no_val:
                val_num = 0
                test_num = int(0.2 * total_num)
            else:
                val_num = int(0.2 * total_num)
            # pl = pseudo_labels
            pl = data.y
            if no_val:
                t_mask, val_mask, te_mask = generate_random_mask(data.x.shape[0], train_num, val_num, test_num)
            else:
                t_mask, val_mask, te_mask = generate_random_mask(data.x.shape[0], train_num, val_num)
            if strategy != 'no':
                train_mask = active_generate_mask(te_mask, data.x, s, data.x.device, strategy, real_budget, data, num_centers, compensation, dataset, logit_path, llm_strategy, pl, max_part, oracle_acc, reliability_list, conf, dataset)
                val_mask = torch.zeros_like(train_mask)
                te_mask = ~train_mask
            else:
                train_mask = t_mask
            new_train_masks.append(train_mask)
            new_val_masks.append(val_mask)
            new_test_masks.append(te_mask)
        ## start from 
    # import ipdb; ipdb.set_trace()
    data.train_masks = new_train_masks
    data.val_masks = new_val_masks
    data.test_masks = new_test_masks
    if split == 'pl_noise_random':
        data = inject_random_noise(data, random_noise)
    else:
        if 'pl' in split or 'active' in split:
            data.ys = ys
    # import ipdb; ipdb.set_trace()
    ## show label performance
    print("Selection complete!")

    entropies = []
    ### do a sanity check for active pl here
    if 'pl' in split or 'active' in split:
        accs = []
        for i in range(len(seeds)):
            # if llm_strategy != 'none':
            #     assert -1 not in data.ys[i][data.train_masks[i]]
            train_mask = data.train_masks[i]
            p_y = data.ys[i]
            y = data.y
            this_acc = (p_y[train_mask] == y[train_mask]).sum().item() / train_mask.sum().item()
            accs.append(this_acc)
            entropies.append(compute_entropy(p_y[train_mask]))
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"Average label accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
        budget = data.train_masks[0].sum()
        print(f"Budget: {budget}")
        mean_entro = np.mean(entropies)
        std_entro = np.std(entropies)
        print(f"Entropy of the labels: {mean_entro:.2f} ± {std_entro:.2f}")

    if 'active' in split and not no_val:
        ## deal with train/val split
        for i in range(len(seeds)):
            train_mask = data.train_masks[i]
            train_idx = torch.where(train_mask)[0]
            val_idx = select_random_indices(train_idx, 0.25)
            new_val_mask = torch.zeros_like(train_mask)
            train_mask[val_idx] = 0
            new_val_mask[val_idx] = 1
            data.val_masks[i] = new_val_mask
            data.train_masks[i] = train_mask
    
    if 'active' in split and random_noise > 0:
        for i in range(len(seeds)):
            non_test_mask = data.train_masks[i] | data.val_masks[i]
            # train_idx = torch.where(train_mask)[0]
            new_y = inject_random_noise_y_level(data.y[non_test_mask], 1 - accs[i])
            data.ys[i][non_test_mask] = new_y
    if save_data:
        torch.save(data, osp.join(data_path, f"{dataset}_{split}_{data_format}.pt"))
    
    print("fuck u")
    if dataset == 'products':
        ## optimize according to OGB
        if osp.exists("../../../ogb/preprocessed_data/aax/products_adj.pt"):
            adj_t = torch.load("../../../ogb/preprocessed_data/aax/products_adj.pt", map_location='cpu')
            data.adj_t = adj_t
        else:
            data = T.ToSparseTensor(remove_edge_index=False)(data)
            data = data.to('cpu')
            ## compute following on cpu
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t
            torch.save(adj_t, "../../../ogb/preprocessed_data/aax/products_adj.pt")
        backup_device = data.x.device
        data = data.to(backup_device)
    # import ipdb; ipdb.set_trace()
    # if train_stage == True:
    #     exit()
    data.conf = conf
    print("Data ready!")
    return data


def active_sort(budget, strategy, x_embed, logits=None, train_mask = None, data = None, seed = None, device = None, num_centers = None, compensation = None, llm_strategy = 'none', pl = None, max_part = 0, oracle_acc = 1, reliability_list = None, confidence = None, name = 'cora', alpha = 0.1, beta = 0.1, gamma = 0.1):
    """
        Given a data obj, return the indices of selected nodes
    """
    # density_query, uncertainty_query, coreset_greedy_query, degree_query, pagerank_query, age_query, cluster_query, gpart_query
    indices = None
    num_of_classes = data.y.max().item() + 1
    if strategy == 'density':
        indices, _ = density_query(budget, x_embed, num_of_classes, train_mask, seed, data, device)
    if strategy == 'density2':
        indices, _ = budget_density_query(budget, x_embed, train_mask, seed, data, device)
    if strategy == 'density3':
        data.params = {}
        data.params['age'] = [alpha]
        indices, _ = budget_density_query2(budget, x_embed, train_mask, seed, data, device)
    elif strategy == 'uncertainty':
        indices = uncertainty_query(budget, logits, train_mask)
    elif strategy == 'coreset':
        indices = coreset_greedy_query(budget, x_embed, train_mask )
    elif strategy == 'degree':
        indices =  degree_query(budget, train_mask, data, device)
    elif strategy == 'degree2':
        data.params = {}
        data.params['age'] = [alpha]
        # import ipdb; ipdb.set_trace()
        indices = degree2_query(budget, x_embed, data, train_mask, seed, device)
        # import ipdb; ipdb.set_trace()
    elif strategy == 'pagerank':
        indices = pagerank_query(budget, train_mask, data, seed)
    elif strategy == 'pagerank2':
        data.params = {}
        data.params['age'] = [alpha]
        indices = pg2_query(budget, x_embed, data, train_mask, seed, device)
    elif strategy == 'age':
        data.params = {}
        data.params['age'] = [alpha]
        indices = age_query(budget, x_embed, data, train_mask, seed, device)
    elif strategy == 'age2':
        data.params = {}
        data.params['age'] = [alpha, beta]
        indices = age_query2(budget, x_embed, data, train_mask, seed, device)
    elif strategy == 'cluster':
        indices = cluster_query(budget, x_embed, data.edge_index, train_mask, seed, device)
    elif strategy == 'cluster2':
        indices = cluster2_query(budget, x_embed, data, data.edge_index, train_mask, seed, device)
    elif strategy == 'gpart':
        data = gpart_preprocess(data, max_part, name)
        indices = gpart_query(budget, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part)
    elif strategy == 'gpart2':
        data = gpart_preprocess(data, max_part, name)
        indices = gpart_query2(budget, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part)
    elif strategy == 'gpartfar':
        data = gpart_preprocess(data, max_part, name)
        indices = gpart_query(budget, num_centers, data, train_mask, 1, x_embed, seed, device, max_part)
    elif strategy == 'llm':
        indices = graph_consistency_for_llm(budget, data, pl, llm_strategy)
    elif strategy == 'random':
        indices = random_query(budget, train_mask, seed)
    elif strategy == 'rim':
        reliability_list.append(torch.ones(data.x.shape[0]))
        indices = rim_query(budget, data.edge_index, train_mask, data, oracle_acc, 0.05, 5, reliability_list, seed)
    elif strategy == 'rim2':
        reliability_list.append(torch.ones(data.x.shape[0]))
        indices = rim_wrapper(budget, data.edge_index, train_mask, data, oracle_acc, 0.15, 1, reliability_list, seed, density_based = True)
        # indices = rim_query(budget, data.edge_index, train_mask, data, oracle_acc, 0.05, 5, reliability_list, seed)
    elif strategy == 'weight':
        reliability_list.append(torch.ones(data.x.shape[0]))
        indices, _ = active_llm_query(b, edge_index, x, confidence, seed, train_mask, data, reliability_list)
    elif strategy == 'hybrid':
        data = gpart_preprocess(data, max_part, name)
        indices = partition_hybrid(budget, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part)
    # elif strategy == 'confidence':
    #     indices = sole_confidence(budget, confidence, train_mask)
    elif strategy == 'featprop':
        # import ipdb; ipdb.set_trace()
        indices = featprop_query(budget, data.x, data.edge_index, train_mask, seed, device)
    elif strategy == 'single':
        indices = single_score(budget, data.x, data, train_mask, seed, confidence, device)
    elif strategy == 'iterative':
        reliability_list.append(torch.ones(data.x.shape[0]))
        indices = iterative_score(budget, data.edge_index, 0.7, train_mask, data, 0.05, reliability_list,  5, seed, device)
    return indices 

def entropy_change(labels, deleted):
    """Return a tensor of entropy changes after removing each label."""
    # Get the counts of each label
    # label_counts = torch.bincount(labels)
    label_list = labels.tolist()
    # valid_labels = labels[train_mask]
    # Compute the original entropy
    original_entropy = compute_entropy(labels)
    mask = torch.ones_like(labels, dtype=torch.bool)

    changes = []
    for i, y in enumerate(labels):
        if deleted[i]:
            changes.append(-np.inf)
            continue    
        if label_list.count(y.item()) == 1:
            changes.append(-np.inf)
            continue
        temp_train_mask = mask.clone()
        temp_train_mask[i] = 0
        # temp_labels = label_list.copy()
        # temp_labels.pop(i)
        v_labels = labels[temp_train_mask]
        new_entropy = compute_entropy(v_labels)
        diff = original_entropy - new_entropy
        changes.append(diff)
    return torch.tensor(changes)



def post_process(train_mask, data, conf, old_y, ratio = 0.3, strategy = 'conf_only'):
    ## conf_only, density_only, conf+density, conf+density+entropy
    # zero_conf = (conf == 0)
    budget = train_mask.sum()
    b = int(budget * ratio)
    num_nodes = train_mask.shape[0]
    num_classes = data.y.max().item() + 1
    N = num_nodes
    labels = data.y
    density_path = osp.join("../../../ogb/preprocessed_data/aax", 'density_x_{}_{}.pt'.format(num_nodes, num_classes))
    density = torch.load(density_path, map_location='cpu')
    if strategy == 'conf_only':
        conf[~train_mask] = 0
        conf_idx_sort = torch.argsort(conf)
        sorted_idx = conf_idx_sort[conf[conf_idx_sort] != 0]
        sorted_idx = sorted_idx[:int(len(sorted_idx) * ratio)]
        train_mask[sorted_idx] = 0
        return train_mask
    elif strategy == 'density_only':
        density[~train_mask] = 0
        density_idx_sort = torch.argsort(density)
        sorted_idx = density_idx_sort[density[density_idx_sort] != 0]
        sorted_idx = sorted_idx[:int(len(sorted_idx) * ratio)]
        train_mask[sorted_idx] = 0
        return train_mask
    elif strategy == 'conf+density':
        conf[~train_mask] = 0
        density[~train_mask] = 0
        oconf = conf.clone()
        odensity = density.clone()
        percentile = (torch.arange(N, dtype=data.x.dtype) / N)
        id_sorted = conf.argsort(descending=True)
        oconf[id_sorted] = percentile
        id_sorted = density.argsort(descending=True)
        odensity[id_sorted] = percentile
        score = oconf + odensity
        score[~train_mask] = 0
        _, indices = torch.topk(score, k=b)
        train_mask[indices] = 0
        return train_mask
    elif strategy == 'conf+density+entropy':
        for _ in range(b):
            conf[~train_mask] = 0
            density[~train_mask] = 0
            # entropy_change = 
            echange = -entropy_change(labels)
            echange[~train_mask] = 0
            percentile = (torch.arange(N, dtype=data.x.dtype) / N)
            id_sorted = conf.argsort(descending=False)
            conf[id_sorted] = percentile
            id_sorted = density.argsort(descending=False)
            density[id_sorted] = percentile
            id_sorted = echange.argsort(descending=False)
            echange[id_sorted] = percentile

            score = conf + density + echange
            score[~train_mask] = np.inf
            idx = torch.argmin(score)
            train_mask[idx] = 0
        return train_mask
    elif strategy == 'conf+entropy':
        # import ipdb; ipdb.set_trace()
        train_idxs = torch.arange(num_nodes)[train_mask]
        mapping = {i: train_idxs[i] for i in range(len(train_idxs))}
        labels = data.y[train_mask]
        s_conf = conf[train_mask]
        selections = []
        N = len(labels)
        deleted = torch.zeros_like(labels)
        for _ in tqdm(range(b)):
            # conf[~train_mask] = 0
            # density[~train_mask] = 0
            # entropy_change = 

            
            echange = -entropy_change(labels, deleted)
            # echange[~train_mask] = -np.inf
            echange[deleted] = np.inf
            percentile = (torch.arange(N, dtype=data.x.dtype) / N)
            id_sorted = s_conf.argsort(descending=True)
            s_conf[id_sorted] = percentile
            # id_sorted = density.argsort(descending=False)
            # density[id_sorted] = percentile
            id_sorted = echange.argsort(descending=True)
            echange[id_sorted] = percentile
            #s_conf[deleted] = np.inf

            score = s_conf + echange
            score[deleted == 1] = np.inf
            idx = torch.argmin(score)
            selections.append(idx)
            deleted[idx] = 1
            #rain_mask[idx] = 0
        for idx in selections:
            train_mask[mapping[idx.item()]] = 0
        return train_mask







    




def compute_entropy(label_tensor):
    # Count unique labels and their occurrences
    unique_labels, counts = label_tensor.unique(return_counts=True)
    # Calculate probabilities
    probabilities = counts.float() / label_tensor.size(0)
    # Compute entropy
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy



def active_generate_mask(test_mask, x, seed, device, strategy, budget, data, num_centers, compensation, dataset_name, path, llm_strategy = 'none', pl = None, max_part = 0, oracle_acc = 1, reliability_list = None, conf  = None, name = 'cora', alpha = 0.1, beta = 0.1, gamma = 0.1):
    """
     x is the initial node feature
     for logits based 
     we first generate logits using random mask (test mask is the same)
    """
    # select_mask = ~test_mask
    select_mask = torch.ones_like(test_mask)
    # idx_select_from = total_idxs[select_mask]
    split = 'random' if dataset_name not in ['arxiv', 'products'] else 'fixed'
    # if strategy == 'uncertainty' or strategy == 'age':
    #     logits = get_logits(dataset_name, split, seed, 'GCN', path)[0]
    # else:
    logits = None
    # import ipdb; ipdb.set_trace()
    active_index = active_sort(budget, strategy, x, logits, select_mask, data, seed, device, num_centers, compensation, llm_strategy, pl, max_part, oracle_acc, reliability_list, confidence=conf, name = name, alpha = alpha, beta = beta, gamma = gamma)
    # import ipdb; ipdb.set_trace()
    new_train_mask = torch.zeros_like(test_mask)
    new_train_mask[active_index] = 1
    return new_train_mask


def get_logits(dataset, split, seed, model, path):
    if not osp.exists(osp.join(path, f"{dataset}_pl_random_{model}_{seed}_logits.pt")):
        return None
    return torch.load(osp.join(path, f"{dataset}_pl_random_{model}_{seed}_logits.pt"), map_location='cpu')




def graph_consistency_for_llm(budget, s_data, gt, filter_strategy = "none"):
    test_node_idxs, train_node_idxs = get_sampled_nodes(s_data)
    one_hop_neighbor_dict = get_one_hop_neighbors(s_data, test_node_idxs)
    two_hop_neighbor_dict = get_two_hop_neighbors_no_multiplication(s_data, test_node_idxs)

    if filter_strategy == "one_hop":
        sorted_nodes = graph_consistency(one_hop_neighbor_dict, gt)
        sorted_nodes = sorted_nodes[::-1]
        return sorted_nodes[:budget]
    elif filter_strategy == "two_hop":
        sorted_nodes = graph_consistency(two_hop_neighbor_dict, gt)
        sorted_nodes = sorted_nodes[::-1]
        return sorted_nodes[:budget]



def gpart_preprocess(data, max_part, name = 'cora'):
    filename = "../../../ogb/preprocessed_data/partitions/{}.pt".format(name)
    if os.path.exists(filename):
        part = torch.load(filename)
        data.partitions = part
    else:
        edge_index = data.edge_index
        data.g = nx.Graph()
        edges = [(i.item(), j.item()) for i, j in zip(edge_index[0], edge_index[1])]
        data.g.add_edges_from(edges)
        graph = data.g.to_undirected()
        graph_part = GraphPartition(graph, data.x, max_part)
        communities = graph_part.clauset_newman_moore(weight=None)
        sizes = ([len(com) for com in communities])
        threshold = 1/3
        if min(sizes) * len(sizes) / len(data.x) < threshold:
            data.partitions = graph_part.agglomerative_clustering(communities)
        else:
            sorted_communities = sorted(communities, key=lambda c: len(c), reverse=True)
            data.partitions = {}
            data.partitions[len(sizes)] = torch.zeros(data.x.shape[0], dtype=torch.int)
            for i, com in enumerate(sorted_communities):
                data.partitions[len(sizes)][com] = i
        torch.save(data.partitions, filename)
    return data



def active_train_mask(test_mask, data, conf, strategy, seed):
    select_mask = ~test_mask
    if strategy == 'random':
        indices = random_query(budget, train_mask, seed)
    elif strategy == 'ours':
        indices = active_llm_query(budget, data.edge_index, data.x, conf, seed)
    new_train_mask = torch.zeros_like(test_mask)
    new_train_mask[indices] = 1
    return new_train_mask