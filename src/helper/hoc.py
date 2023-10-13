import torch
import torch.nn.functional as F

from utils import *


def get_T_HOC(config, model, train_dataloader_EF, rnd, test_flag=False, max_step=501, T0=None, p0=None, lr=0.1):
    config.path, record, c1m_cluster_each = init_feature_set(config, model, train_dataloader_EF, rnd)
    sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, c1m_cluster_each)

    if test_flag:
        return 0, 0, 0

    if config.loss == 'fw':  # forward loss correction
        # return one matrix if global
        # return a set of matrices + a map between index and matrix
        T_est, P_est, T_init, T_err = get_T_P_global(config, sub_noisy_dataset_name, max_step, T0, p0, lr=lr)
        # T_est = config.T
        if config.local:
            T_local, map_index_T, T_err = get_T_P_local(config, sub_noisy_dataset_name, T_est)
            return T_local, map_index_T, T_err
        else:
            return T_est, T_init, T_err
    else:
        return 0, 0, 0


def get_T_P_global(config, sub_noisy_dataset_name, max_step=501, T0=None, p0=None, lr=0.1):
    global GLOBAL_T_REAL
    # all_point_cnt = 10000
    all_point_cnt = 15000
    # all_point_cnt = 2000
    NumTest = int(50)
    # NumTest = int(20)
    # TODO: make the above parameters configurable

    print(f'Estimating global T. Sampling {all_point_cnt} examples each time')

    KINDS = config.num_classes
    data_set = torch.load(f'{sub_noisy_dataset_name}', map_location=torch.device('cpu'))
    T_real, P_real = check_T_torch(KINDS, data_set['clean_label'], data_set['noisy_label'])
    GLOBAL_T_REAL = T_real
    p_real = count_real(KINDS, torch.tensor(T_real), torch.tensor(P_real), -1)

    # Build Feature Clusters --------------------------------------
    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True)

        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
            ss = torch.abs(p_estimate[i] / (idx + 1) - p_real[i])
            p_estimate_rec[idx, i] = torch.mean(torch.abs(p_estimate[i] / (idx + 1) - p_real[i])) * 100.0 / (
                torch.mean(p_real[i]))  # Assess the gap between estimation value and real value
        print(p_estimate_rec[idx], flush=True)

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, config.device, max_step, T0, p0, lr=lr)
    P_calc = P_calc.view(-1).cpu().numpy()
    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    # print("----Real value----------")
    # print(f'Real: P = {P_real},\n T = \n{np.round(np.array(T_real),3)}')
    # print(f'Sum P = {sum(P_real)},\n sum T = \n{np.sum(np.array(T_real), 1)}')
    # print("\n----Calc result----")
    # print(f"loss = {loss_min}, \np = {P_calc}, \nT_est = \n{np.round(E_calc, 3)}")
    # print(f"sum p = {np.sum(P_calc)}, \nsum T_est = \n{np.sum(E_calc, 1)}")
    # print("\n---Error of the estimated T (sum|T_est - T|/N * 100)----", flush=True)
    print(f"L11 Error (Global): {np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / KINDS * 100}")
    T_err = np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / KINDS * 100
    rec_global = [[] for _ in range(3)]
    rec_global[0], rec_global[1], rec_global[2] = loss_min, T_real, E_calc
    path = "./rec_global/" + config.dataset + "_" + config.label_file_path[11:14] + "_" + config.pre_type + ".pt"
    torch.save(rec_global, path)
    return E_calc, P_calc, T_init, T_err


def get_T_P_local(config, sub_noisy_dataset_name, T_avg=None):
    global GLOBAL_T_REAL
    rounds = 300
    all_point_cnt = 100  # 500 for global  100 for local
    NumTest = int(30)
    # TODO: make the above parameters configurable

    print(f'Estimating local T. Sampling {all_point_cnt} examples each time')
    KINDS = config.num_classes
    data_set = torch.load(f'{sub_noisy_dataset_name}', map_location=torch.device('cpu'))

    next_select_idx = np.random.choice(range(data_set['index'].shape[0]), 1, replace=False)
    selected_idx = torch.tensor(range(data_set['index'].shape[0]))
    T_rec = []
    T_true_rec = []
    map_index_T = np.zeros((data_set['index'].shape[0]), dtype='int') - 1

    round = 0
    T_err_list = []
    while (1):
        # for round in range(rounds):

        if config.local:  # Select a picture & nearest 250 pictures: count t_ Real and P_ real

            # Start a cycle here and run about 300 * numtest times to the end
            # One center is extracted each time, and numLocal adjacent points are taken as cluster, which are recorded as selected_ idx
            idx_sel = torch.tensor(
                extract_sub_dataset_local(data_set['feature'], next_select_idx, numLocal=config.numLocal))
            # assign round value to the locations with value -1
            map_index_T[idx_sel[map_index_T[idx_sel] == -1]] = round
            next_select_idx, selected_idx = select_next_idx(selected_idx, idx_sel)
            T_real, P_real = check_T_torch(KINDS, data_set['clean_label'][idx_sel],
                                           data_set['noisy_label'][idx_sel])  # focus on 250 samples
            # T and P of local cluster
        p_real = count_real(KINDS, torch.tensor(T_real), torch.tensor(P_real), -1) if config.local else count_real(
            KINDS, torch.tensor(config.T), torch.tensor(config.P), -1)

        # Build Feature Clusters --------------------------------------
        p_estimate = [[] for _ in range(3)]
        p_estimate[0] = torch.zeros(KINDS)
        p_estimate[1] = torch.zeros(KINDS, KINDS)
        p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
        p_estimate_rec = torch.zeros(NumTest, 3)
        for idx in range(NumTest):

            # local
            sample = np.random.choice(idx_sel, all_point_cnt,
                                      replace=False)  # test: extract 100 samples from local cluster
            final_feat = data_set['feature'][sample]
            noisy_label = data_set['noisy_label'][sample]
            #
            cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
            for i in range(3):
                cnt_y_3[i] /= all_point_cnt
                p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
                p_estimate_rec[idx, i] = torch.mean(torch.abs(p_estimate[i] / (idx + 1) - p_real[i])) * 100.0 / (
                    torch.mean(p_real[i]))

        # Calculate T & P  -------------------------------------------------------------
        for j in range(3):
            p_estimate[j] = p_estimate[j] / NumTest

        loss_min, E_calc, P_calc, _ = calc_func(KINDS, p_estimate, True,
                                                config.device)  # E_calc, P_calc = calc_func(p_real)
        P_calc = P_calc.view(-1).cpu().numpy()
        E_calc = E_calc.cpu().numpy()

        center_label = np.argmax(P_real)
        T_rec += [P_calc.reshape(-1, 1) * E_calc + (1 - P_calc).reshape(-1, 1) * T_avg]  # estimated local T
        T_true_rec += [P_real.reshape(-1, 1) * T_real + (1 - P_real).reshape(-1, 1) * GLOBAL_T_REAL]

        # print("\n---Error of the estimated T (sum|T_est - T|/N * 100)----", flush=True)
        # print("----T_rec[round]", np.array(T_rec[round]))
        # print("----T_true_rec[round]", np.array(T_true_rec[round]))
        T_err = np.sum(np.abs(np.array(T_rec[round]) - np.array(T_true_rec[round]))) * 1.0 / KINDS * 100
        print(f"L11 Error (Local): {T_err}")
        T_err_list.append(T_err)

        print(f'round {round}, remaining {np.sum(map_index_T == -1)}')
        round += 1
        if round == rounds:
            print(
                f'Only get local transition matrices for the first {np.sum(map_index_T != -1)} examples in {rounds} rounds',
                flush=True)
            T_rec += [T_avg]
            map_index_T[map_index_T == -1] = round  # use T_avg for the remaining matrices
            return T_rec, map_index_T, T_err_list
        if selected_idx[selected_idx > -1].size(0) == 0:
            return T_rec, map_index_T, T_err_list
            # did not use P currently


def func(KINDS, p_estimate, T_out, P_out, N, step, LOCAL, _device):
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)  # define the loss

    P = smp(P_out)
    T = smt(T_out)

    mode = random.randint(0, KINDS - 1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N, N*N*N
    p_temp = count_real(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    weight = [1.0, 1.0, 1.0]
    # weight = [2.0,1.0,1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j])  # / np.sqrt(N**j)

    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P + eps)) / 10

    return loss


def calc_func(KINDS, p_estimate, LOCAL, _device, max_step=501, T0=None, p0=None, lr=0.1):
    # init
    # _device =  torch.device("cpu")
    N = KINDS
    eps = 1e-8
    if T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N, N))
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device=None) / N + torch.rand((N, 1), device=None) * 0.1  # P：0-9 distribution
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr=lr)

    # train
    loss_min = 100.0
    T_rec = T.detach()
    P_rec = P.detach()

    time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(KINDS, p_estimate, T, P, N, step, LOCAL, _device)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        # if step % 10 == 0:
        #     print('loss {}'.format(loss))
        #     print(f'step: {step}  time_cost: {time.time() - time1}')
        #     print(f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
        #     print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
        #     time1 = time.time()
    # if global_var.get_value('T_init') is None:
    global_var.set_value('T_init', T_rec.detach())
    # tmp = global_var.get_value('T_init')
    # print(f'set T_init to {tmp}')
    # if global_var.get_value('p_init') is None:
    global_var.set_value('p_init', P_rec.detach())
    print(f'T_init and p_init are updated')
    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def count_y(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    max_val = np.max(dist)
    am = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist, axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt


def count_2nn_acc(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    # print(dist.shape)
    # print(f'Use Euclidean distance')
    # dist = distEuclidean(feat_cord, feat_cord)

    max_val = np.max(dist)
    am = np.argmin(dist, axis=1)
    # TODO: speedup this part
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist, axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt

def count_knn_conf(args, feat_cord, label, cluster_sum, k):
    # feat_cord = torch.tensor(final_feat)
    KINDS = args.num_classes
    # cnt = [[] for _ in range(3)]
    # cnt[0] = torch.zeros(KINDS)
    # cnt[1] = torch.zeros(KINDS, KINDS)
    # cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    dist = cosDistance(feat_cord)

    print(f'knn parameter is k = {k}')
    time1 = time.time()
    min_similarity = args.min_similarity
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    knn_labels = label[indices]
    knn_labels_cnt = torch.zeros(cluster_sum, KINDS)

    for i in range(KINDS):
        knn_labels_cnt[:, i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i),
                                          1)  # similarity should be larger than min_similarity
        # print(knn_labels_cnt[0])
    time2 = time.time()
    print(f'Running time for k = {k} is {time2 - time1}')
    confidence = torch.sum(knn_labels_cnt, 1).reshape(-1)
    return confidence


def count_knn_distribution(args, feat_cord, label, cluster_sum, k, norm='l2'):
    # feat_cord = torch.tensor(final_feat)
    KINDS = args.num_classes
    # cnt = [[] for _ in range(3)]
    # cnt[0] = torch.zeros(KINDS)
    # cnt[1] = torch.zeros(KINDS, KINDS)
    # cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    dist = cosDistance(feat_cord)
    # dist = torch.cdist(feat_cord,feat_cord,p=2)
    # import pdb
    # pdb.set_trace()

    print(f'knn parameter is k = {k}')
    time1 = time.time()
    min_similarity = args.min_similarity
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    values[:, 0] = 2.0 * values[:, 1] - values[:, 2]
    knn_labels = label[indices]

    # # check knn feasibility
    # feasibility = torch.zeros(k)
    # prob = torch.zeros(k)
    # # import pdb
    # # pdb.set_trace()
    # e=0.4
    # import scipy.special as sc
    # for i in range(k):
    #     feasibility[i] = torch.mean(1.0*(torch.sum(knn_labels[:,:i+1] == knn_labels[:,0].view(50000,1),1) == i+1))
    #     k1=int(np.ceil((i+1)/2)-1)
    #     a = sc.betainc(i-k1+1,k1+1,1-e)
    #     prob[i] = feasibility[i] * a

    # print(f'delta_k is: {1-feasibility}')
    # print(f'probability lower bound is: {prob}')
    # torch.save({'delta_k': 1-feasibility, 'prob': prob}, f'{args.pre_type}_c100_{k}.pt')

    # # # e=0.4
    # # # k=20
    # # # k1=int(np.ceil((k+1)/2)-1)
    # # # a = sc.betainc(k-k1+1,k1+1,1-e)

    # # # k=5
    # # # k1=int(np.ceil((k+1)/2)-1)
    # # # b = sc.betainc(k-k1+1,k1+1,1-e)
    # # # # b/a
    # # # print(f'a={a}, b={b}, b/a={a/b}')
    # exit()

    knn_labels_cnt = torch.zeros(cluster_sum, KINDS)
    # thre_val_tmp = torch.tensor([[0.1766, 0.2208],
    #         [0.1423, 0.1991],
    #         [0.1439, 0.1672],
    #         [0.1063, 0.1464],
    #         [0.1192, 0.1708],
    #         [0.1318, 0.1464],
    #         [0.1206, 0.1672],
    #         [0.1151, 0.1795],
    #         [0.1452, 0.2184],
    #         [0.1517, 0.1991]])
    # thre_val = torch.mean(thre_val_tmp,1)

    for i in range(KINDS):
        # knn_labels_cnt[:,i] += torch.sum(1.0 * (knn_labels == i), 1)
        knn_labels_cnt[:, i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i), 1)
        # knn_labels_cnt[:,i] += torch.sum((1.0 - min_similarity - values) * (knn_labels == i) * (values < thre_val[i]), 1)  # similarity should be larger than min_similarity
        # print(knn_labels_cnt[0])
    time2 = time.time()
    print(f'Running time for k = {k} is {time2 - time1}')

    # # ----------- old -------------
    # # feat_cord = feat_cord.cpu().numpy()
    # # dist = distCosine(feat_cord, feat_cord)
    # # import pdb
    # # pdb.set_trace()

    # # print(f'Use Euclidean distance')
    # # dist = distEuclidean(feat_cord, feat_cord)

    # max_val = np.max(dist)
    # k += 1 # k-nn -> k+1 instances
    # knn_labels_cnt = torch.zeros(cluster_sum, KINDS)
    # for k_loop in range(k):
    #     # print(k_loop, flush=True)
    #     min_dis_id = np.argmin(dist,axis=1)
    #     knn_labels = label[min_dis_id]

    #     # not count self label
    #     # if k_loop > 0:
    #     #     for i in range(cluster_sum):
    #     #         knn_labels_cnt[i, knn_labels[i]] += 1
    #     #         dist[i][min_dis_id[i]] = 10000.0 + max_val
    #     # else:
    #     #     for i in range(cluster_sum):
    #     #         dist[i][min_dis_id[i]] = 10000.0 + max_val

    #     # # count self label
    #     # for i in range(cluster_sum):
    #     #     knn_labels_cnt[i, knn_labels[i]] += 1
    #     #     dist[i][min_dis_id[i]] = 10000.0 + max_val

    #     # count self label, distance as weights (weight = 1-dist)
    #     for i in range(cluster_sum):
    #         knn_labels_cnt[i, knn_labels[i]] += 1-dist[i][min_dis_id[i]]
    #         dist[i][min_dis_id[i]] = 10000.0 + max_val

    if norm == 'l2':
        # normalized by l2-norm -- cosine distance
        knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
    elif norm == 'l1':
        # normalized by mean
        knn_labels_prob = knn_labels_cnt / torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
    else:
        raise NameError('Undefined norm')
    return knn_labels_prob


def get_score(knn_labels_cnt, label, k, method='cores', prior=None):  # method = ['cores', 'peer']
    # knn_labels_cnt: sampleSize * #class
    # knn_labels_cnt /= (k*1.0)
    # import pdb
    # pdb.set_trace()
    loss = F.nll_loss(torch.log(knn_labels_cnt + 1e-8), label, reduction='none')
    # loss = -torch.tanh(-F.nll_loss(knn_labels_cnt, label, reduction = 'none')) # TV
    # loss = -(-F.nll_loss(knn_labels_cnt, label, reduction = 'none')) # 
    # loss_numpy = loss.data.cpu().numpy()
    # num_batch = len(loss_numpy)
    # loss_v = np.zeros(num_batch)
    # loss_div_numpy = float(np.array(0))

    # loss_ = -(knn_labels_cnt)   # 
    # loss_ = -torch.tanh(knn_labels_cnt)   # TV 
    # import pdb
    # pdb.set_trace()
    loss_ = -torch.log(knn_labels_cnt + 1e-8)
    if method == 'cores':
        score = loss - torch.mean(loss_, 1)
        # score =  loss
    elif method == 'peer':
        prior = torch.tensor(prior)
        score = loss - torch.sum(torch.mul(prior, loss_), 1)
    elif method == 'ce':
        score = loss
    elif method == 'avg':
        score = - torch.mean(loss_, 1)
    elif method == 'new':
        score = 1.1 * loss - torch.mean(loss_, 1)
    else:
        raise NameError('Undefined method')

    return score