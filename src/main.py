from helper.utils import load_yaml, pkl_and_write
from helper.args import get_command_line_args, replace_args_with_dict_values
from helper.noisy import NoiseAda
import torch
from helper.train_utils import train, test, get_optimizer, seed_everything, s_train, batch_train, batch_test
from models.nn import get_model
import numpy as np
import time
import logging
# print("OK")
import torch.nn.functional as F
from copy import deepcopy
from helper.data import get_dataset
import os.path as osp
import optuna
from helper.hyper_search import hyper_search
import sys
from tqdm import tqdm
# from helper.utils import delete_non_tensor_attributes
# from ogb.nodeproppred import Evaluator


# def train_pipeline_batch(seeds, args, epoch, data, writer, need_train, mode="main"):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     test_result_acc = []
#     early_stop_accum = 0
#     val_result_acc = []
#     out_res = []
#     best_val = 0
#     evaluator = Evaluator(name='ogbn-products')
#     if args.inductive:
#         data = to_inductive(data)
#     if mode == "main":
#         split_num = args.num_split
#     else:
#         split_num = args.sweep_split
#     split = 0
#     data.train_mask = data.train_masks[split]
#     data.val_mask = data.val_masks[split]
#     data.test_mask = data.test_masks[split]
#     data = delete_non_tensor_attributes(data)
#     assert split_num == 1
#     for seed in seeds:
#         set_seed_config(seed)
#         model = get_model(args).to(device)
#         optimizer, scheduler = get_optimizer(args, model)
#         loss_fn = torch.nn.CrossEntropyLoss()
#         best_val = 0
#         for split in range(split_num):
#             if args.normalize:
#                 data.x = F.normalize(data.x, dim = -1)
#             input_nodes = torch.arange(data.x.shape[0])[data.train_mask]
#             # import ipdb; ipdb.set_trace()
#             data = data.to(device, 'x', 'y')
#             subgraph_loader = NeighborLoader(data, input_nodes=input_nodes,
#                                 num_neighbors=[15, 10, 5],
#                     batch_size=1024, shuffle=True,
#                     num_workers=4)
#             val_loader = NeighborLoader(data, input_nodes=None, batch_size=4096, shuffle=False,
#                                 num_neighbors=[-1], num_workers=1, persistent_workers=True)
#             # import ipdb; ipdb.set_trace()
#             for epoch in range(1, args.epochs + 1):
#                 train_loss = batch_train(model, subgraph_loader, optimizer, device)
#                 if scheduler:
#                     scheduler.step()
#                 val_acc = batch_test(model, data, evaluator, val_loader, device, data.val_mask)
#                 print(f"Epoch {epoch}: Train loss: {train_loss}, Val acc: {val_acc}")
#                 if val_acc > best_val:
#                     best_val = val_acc
#                     best_model = deepcopy(model)
#                     early_stop_accum = 0
#                 else:
#                     if epoch >= args.early_stop_start:
#                         early_stop_accum += 1
#                     if early_stop_accum > args.early_stopping and epoch >= args.early_stop_start:
#                         break
#             test_acc = batch_test(model, data, evaluator, val_loader, device, data.test_mask)
#             val_result_acc.append(val_acc)
#             test_result_acc.append(test_acc)
#     return test_result_acc, val_result_acc




def train_pipeline(seeds, args, epoch, data, need_train, need_save_logits, reliability_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    debug_accs = []
    train_accs = []
    num_of_classes = data.y.max().item() + 1
    if args.model_name == 'S_model':
        noise_ada = NoiseAda(num_of_classes).to(device)
    else:
        noise_ada = None
    for i, seed in enumerate(seeds):
        if len(reliability_list) > 0:
            reliability = reliability_list[0].to(device)
        seed_everything(seed)
        model = get_model(args).to(device)
        optimizer, scheduler = get_optimizer(args, model)
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)            
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')            
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y.clone()
            if not args.debug_gt_label:
                data.y = data.ys[i]
            else:
                print("Using ground truth label")

            # import ipdb; ipdb.set_trace()
        for i in tqdm(range(epoch)):
            # ipdb.set_trace()
            train_mask = data.train_mask
            val_mask = data.val_mask
            if need_train:
                if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                    train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                else:
                    if args.model_name == 'S_model':
                        train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                    else:
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                if scheduler:
                    scheduler.step()
                if args.output_intermediate and not args.no_val:
                    print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                if args.debug:
                    if args.filter_strategy == 'none':
                        test_acc, res = test(model, data, 0, data.test_mask)
                    else:
                        test_acc, res = test(model, data, 0, data.test_mask, data.backup_y)
                    # print(f"Epoch {i}: Test acc: {test_acc}")
                    debug_acc.append(test_acc)
                    this_train_acc.append(train_acc)
                if not args.no_val:
                    if val_acc > best_val:
                        best_val = val_acc
                        best_model = deepcopy(model)
                        early_stop_accum = 0
                    else:
                        if i >= args.early_stop_start:
                            early_stop_accum += 1
                        if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                            print(f"Early stopping at epoch {i}")
                            break
            else:
                best_model = model
        if 'pl' in args.split or 'active' in args.split:
            data.y = data.backup_y
        if args.no_val or best_model == None:
            best_model = model
        test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
        test_result_acc.append(test_acc)
        val_result_acc.append(best_val)
        out_res.append(res)
        best_val = 0
        best_model = None
        if args.debug:
            debug_accs.append(debug_acc)
            train_accs.append(this_train_acc)
        if need_save_logits:
            torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
    if not args.debug:
        return test_result_acc, val_result_acc, out_res
    else:
        return test_result_acc, val_result_acc, out_res, debug_accs, train_accs






def main(data_path, args = None, custom_args = None, save_best = False):
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if custom_args != None:
        args = replace_args_with_dict_values(args, custom_args)
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        if not args.debug:
            test_result_acc, _, _ = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
        else:
            test_result_acc, _, _, debug_accs, train_accs = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
        mean_test_acc = np.mean(test_result_acc) * 100
        std_test_acc = np.std(test_result_acc) * 100
        if args.debug:
            best_possible_test_acc = [np.max(res) for res in debug_accs]
        res_train_accs = [x[-1] for x in train_accs]
        print(f"Train Accuracy: {np.mean(res_train_accs) * 100:.2f} ± {np.std(res_train_accs) * 100:.2f}")
        print(f"Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
        if args.debug:
            print(f"Best possible accuracy: {np.mean(best_possible_test_acc) * 100:.2f} ± {np.std(best_possible_test_acc) * 100:.2f}")
        print("Test acc: {}".format(test_result_acc))
    elif args.ensemble_string != "":
        pass
    else:
        pass
    if save_best:
        pkl_and_write(args, osp.join("./bestargs", f"{args.model_name}_{args.dataset}_{args.data_format}.pkl"))
    if args.debug:
        if args.debug_gt_label:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_gt.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_gt.pkl"))
        elif args.filter_all_wrong_labels:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_filtered.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_filtered.pkl"))
        else:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs.pkl"))


                
def max_trial_callback(study, trial, max_try):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    n_total_complete = len([t for t in study.trials])
    if n_complete >= max_try or n_total_complete >= 2 * max_try:
        study.stop()
        torch.cuda.empty_cache()


def sweep(data_path, args = None):
    # test_seeds = [i for i in range(args.seed_num)]
    sweep_seeds = [i for i in range(args.sweep_seed_num)]
    ## get default command line args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset}_{args.model_name}_{args.data_format}_{args.split}"
    study = optuna.create_study(study_name=study_name, storage=None, direction='maximize', load_if_exists=True)
    param_f = hyper_search
    sweep_round = args.sweep_round
    study.optimize(lambda trial: sweep_run(trial, args, sweep_seeds, param_f, device, data_path), catch=(RuntimeError,), n_trials=sweep_round, callbacks=[lambda study, trial: max_trial_callback(study, trial, sweep_round)], show_progress_bar=True, gc_after_trial=True)
    main(args=args, custom_args = study.best_trial.params, save_best = True)
    print(study.best_trial.params)



def sweep_run(trial, args, sweep_seeds, param_f, device, data_path):
    params = param_f(trial, args.data_format, args.model_name, args.dataset)    
    args = replace_args_with_dict_values(args, params)
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(sweep_seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        test_result_acc, _, _ = train_pipeline(sweep_seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
    elif args.ensemble_string != "":
        pass
    else:
        pass
    mean_test_acc = np.mean(test_result_acc)
    std_test_acc = np.std(test_result_acc)
    print(f"Test Accuracy: {mean_test_acc} ± {std_test_acc}")
    return mean_test_acc






    


    
if __name__ == '__main__':
    current_time = int(time.time())
    # #logging.basicConfig(filename='../../logs/{}.log'.format(current_time),
    #                 filemode='a',
    #                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                 datefmt='%H:%M:%S',
    #                 level=logging.INFO)

    print("Start")

    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)
    data_path = params_dict['DATA_PATH']
    if args.mode == "main":
        main(data_path, args = args)
    else:
        sweep(data_path, args = args)



