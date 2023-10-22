import torch 
import numpy as np
import sklearn.metrics as metrics
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import torch.nn.functional as F
from sklearn import cluster
from torch_geometric.utils import degree as degree_cal
import networkx as nx
import torch_sparse
from torch_sparse import SparseTensor, spmm
import copy
from helper.train_utils import seed_everything
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.nn import LabelPropagation
import numpy as np
import warnings
from models.nn import LinearRegression
import torch.nn as nn
import kmedoids
from torch_geometric.utils import remove_self_loops, scatter, add_self_loops
from sklearn.metrics.pairwise import euclidean_distances
import json 
import codecs
import os
import os.path as osp
# import numba


# import faiss
# from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
warnings.simplefilter("ignore")

# import faiss

GLOBAL_RESULT_PATH = "xxx/data/aax"

class Cluster:
    """
    Kmeans Clustering
    """
    def __init__(self, n_clusters, n_dim, seed,
                 implementation='sklearn',
                 init='k-means++',
                 device=torch.cuda.is_available()):

        assert implementation in ['sklearn', 'faiss', 'cuml']
        assert init in ['k-means++', 'random']

        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.implementation = implementation
        self.initialization = init
        self.model = None

        if implementation == 'sklearn':
            self.model = cluster.KMeans(n_clusters=n_clusters, init=init, random_state=seed)
        else:
            self.model = faiss.Kmeans(self.n_dim, self.n_clusters, niter=20, verbose=True)

    def train(self, x):
        if self.implementation == 'sklearn':
            self.model.fit(x)
        else:
            self.model.train(x)

    def predict(self, x):
        if self.implementation == 'sklearn':
            return self.model.predict(x)
        else:
            return self.model.index.search(x, 1)[1].squeeze()

    def get_centroids(self):
        if self.implementation == 'sklearn':
            return self.model.cluster_centers_
        else:
            return self.model.centroids

    def get_inertia(self):
        if self.implementation == 'sklearn':
            return self.model.inertia_
        else:
            return self.model.obj[-1]


def density_query(b, x_embed, n_cluster, train_mask, seed, data, device):
    # Get propagated nodes
    # Perform K-Means as approximation
    seed_everything(seed)
    num_nodes = x_embed.shape[0]
    cache_path = osp.join(GLOBAL_RESULT_PATH, 'density_x_{}_{}.pt'.format(num_nodes, n_cluster))
    if os.path.exists(cache_path):
        density = torch.load(cache_path, map_location='cpu')
    else:
        kmeans = Cluster(n_clusters=n_cluster, n_dim=x_embed.shape[1], seed=seed, device=device)
        kmeans.train(x_embed)

        # Calculate density
        centers = kmeans.get_centroids()
        label = kmeans.predict(x_embed)
        centers = centers[label]
        dist_map = torch.linalg.norm(x_embed - centers, dim=1)
        dist_map = torch.tensor(dist_map, dtype=x_embed.dtype, device=x_embed.device)
        density = 1 / (1 + dist_map)
        torch.save(density, cache_path)
    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0
    # density[np.where(train_mask == 0)[0]] = -100000
    _, indices = torch.topk(density, k=b)

    return indices, density



def budget_density_query(b, x_embed, train_mask, seed, data, device):
    density = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, x_embed.shape[0], b))

    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0

    _, indices = torch.topk(density, k=b)

    return indices, density

def budget_density_query2(b, x_embed, train_mask, seed, data, device):
    density = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, x_embed.shape[0], b))
    N = x_embed.shape[0]


    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0

    percentile = (torch.arange(N, dtype=x_embed.dtype, device=device) / N)
    id_sorted = density.argsort(descending=False)
    density[id_sorted] = percentile

    n_classes = data.y.max().item() + 1
    density2 = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, x_embed.shape[0], n_classes))
    if hasattr(data, 'drop_idx'):
        density2[data.drop_idx] = 0
    id_sorted = density2.argsort(descending=False)
    density2[id_sorted] = percentile

    alpha = data.params['age'][0]

    score = alpha * density + (1 - alpha) * density2

    _, indices = torch.topk(score, k=b)


    return indices, None



def random_query(b, train_mask, seed):
    seed_everything(seed)
    # indices = list(np.where(train_mask != 0)[0])
    indices = torch.randperm(train_mask.shape[0])
    # np.random.shuffle(indices)
    return indices[:b]


def uncertainty_query(b, logits, train_mask):
    entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
    entropy[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(entropy, k=b)
    return indices


def coreset_greedy_query(b, embed, train_mask):
    indices = []
    for i in range(b):
        dist = metrics.pairwise_distances(embed, embed[indices], metric='euclidean')
        min_distances = torch.min(torch.tensor(dist), dim=1)[0]
        new_index = min_distances.argmax()
        indices.append(int(new_index))
    return indices

def degree_query(b, train_mask, data, device):
    row_index = data.edge_index[0]
    degree = degree_cal(row_index, num_nodes=data.x.shape[0], dtype=torch.long)
    # degree[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(degree, k=b)
    return indices

def pagerank_query(b, train_mask, data, seed):
    seed_everything(seed)
    edges = [(int(i), int(j)) for i, j in zip(data.edge_index[0], data.edge_index[1])]
    nodes = list(range(data.x.shape[0]))
    data.g = nx.Graph()
    data.g.add_nodes_from(nodes)
    data.g.add_edges_from(edges)
    page = torch.tensor(list(pagerank(data.g).values()))
    # page[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(page, k=b)
    return indices



def compute_pagerank(data, num_nodes, device):
    if osp.exists("{}/page_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes)):
        page = torch.load("{}/page_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes))
        return page 
    else:
        edges = [(int(i), int(j)) for i, j in zip(data.edge_index[0], data.edge_index[1])]
        nodes = list(range(data.x.shape[0]))
        data.g = nx.Graph()
        data.g.add_nodes_from(nodes)
        data.g.add_edges_from(edges)
        page = torch.tensor(list(pagerank(data.g).values()), dtype = data.x.dtype, device=device)
        torch.save(page, "{}/page_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes))
        return page


def degree2_query(b, x, data, train_mask, seed, device):
    seed_everything(seed)
    num_nodes = x.shape[0]
    N = num_nodes
    ## get aax
    # aax = compute_norm_aax(x, data.edge_index, num_nodes)
    # page = compute_pagerank(data, num_nodes, device)
    # N = x.shape[0]

    row_index = data.edge_index[0]
    degree = degree_cal(row_index, num_nodes=data.x.shape[0], dtype=x.dtype)
    density_path = osp.join(GLOBAL_RESULT_PATH, 'density_x_{}_{}.pt'.format(num_nodes, b))
    density = torch.load(density_path, map_location='cpu')
    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0
    # Get percentile
    percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
    id_sorted = density.argsort(descending=False)
    density[id_sorted] = percentile
    id_sorted = degree.argsort(descending=False)
    degree[id_sorted] = percentile
    # Get linear combination
    alpha= data.params['age'][0]
    age_score = alpha * density + (1 - alpha) * degree
    # age_score[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(age_score, k=b)
    return indices


def pg2_query(b, x, data, train_mask, seed, device):
    seed_everything(seed)
    num_nodes = x.shape[0]
    ## get aax
    # aax = compute_norm_aax(x, data.edge_index, num_nodes)
    page = compute_pagerank(data, num_nodes, device)
    N = x.shape[0]

    density_path = osp.join(GLOBAL_RESULT_PATH, 'density_x_{}_{}.pt'.format(num_nodes, b))
    density = torch.load(density_path, map_location='cpu')
    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0
    # Get percentile
    percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
    id_sorted = page.argsort(descending=False)
    page[id_sorted] = percentile

    id_sorted = density.argsort(descending=False)
    density[id_sorted] = percentile

    # Get linear combination
    alpha= data.params['age'][0]
    age_score = alpha * density + (1 - alpha) * page
    # age_score[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(age_score, k=b)
    return indices




def age_query(b, x, data, train_mask, seed, device):
    seed_everything(seed)
    num_nodes = x.shape[0]
    ## get aax
    # aax = compute_norm_aax(x, data.edge_index, num_nodes)
    page = compute_pagerank(data, num_nodes, device)
    N = x.shape[0]

    density_path = osp.join(GLOBAL_RESULT_PATH, 'density_aax_{}_{}.pt'.format(num_nodes, b))
    aax_density = torch.load(density_path, map_location='cpu')
    # Get percentile
    percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
    id_sorted = page.argsort(descending=False)
    page[id_sorted] = percentile

    id_sorted = aax_density.argsort(descending=False)
    aax_density[id_sorted] = percentile

    # Get linear combination
    alpha= data.params['age'][0]
    age_score = alpha * aax_density + (1 - alpha) * page
    # age_score[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(age_score, k=b)
    return indices


def age_query2(b, x, data, train_mask, seed, device):
    seed_everything(seed)
    num_nodes = x.shape[0]
    num_classes = data.y.max().item() + 1
    ## get aax
    # aax = compute_norm_aax(x, data.edge_index, num_nodes)
    page = compute_pagerank(data, num_nodes, device)
    N = x.shape[0]

    density_path = osp.join(GLOBAL_RESULT_PATH, 'density_aax_{}_{}.pt'.format(num_nodes, b))
    aax_density = torch.load(density_path, map_location='cpu').to(device)

    d_path = osp.join(GLOBAL_RESULT_PATH, 'density_x_{}_{}.pt'.format(num_nodes, num_classes))
    density = torch.load(d_path, map_location='cpu').to(device)
    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0

    # Get percentile
    percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
    id_sorted = page.argsort(descending=False)
    page[id_sorted] = percentile

    id_sorted = aax_density.argsort(descending=False)
    aax_density[id_sorted] = percentile

    # Get linear combination
    alpha, beta = data.params['age']
    age_score = alpha * aax_density + beta * page + (1 - alpha - beta) * density
    # age_score[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(age_score, k=b)
    return indices





def featprop_query(b, x, edge_index, train_mask, seed, device):
    seed_everything(seed)
    num_nodes = x.shape[0]
    # new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    # adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
    #                sparse_sizes=(num_nodes, num_nodes))
    # adj_matrix2 = adj.matmul(adj)
    # aax = adj_matrix2.matmul(x)
    # aax_dense = aax.to_dense()
    if not osp.exists("{}/km_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b)):
        aax_dense = compute_norm_aax(x, edge_index, num_nodes)
        # aax_dense[np.where(train_mask == 0)[0]] = 0
        distmat = euclidean_distances(aax_dense.cpu().numpy())
        km = kmedoids.fasterpam(distmat, b)
    else:
        km = torch.load("{}/km_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b), map_location='cpu')

    # c = km.fit(aax_dense)
    # indices = c.medoid_indices_
    selected = torch.tensor(np.array(km.medoids, dtype = np.int32))
    select_mask = torch.zeros_like(train_mask)
    select_mask[selected] = 1
    total_idxs = torch.arange(num_nodes)
    # import ipdb; ipdb.set_trace()
    return total_idxs[select_mask]










def cluster2_query(b, x, data, edge_index, train_mask, seed, device):
    seed_everything(seed)
    # Perform K-Means clustering:
    num_nodes = x.shape[0]
    _, density = density_query(b, x, data.y.max().item() + 1, train_mask, seed, device)
    if osp.exists("{}/center_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b)):
        centers = torch.load("{}/center_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b))
    else:
        aax = compute_norm_aax(x, edge_index, num_nodes)
        kmeans = Cluster(n_clusters=b, n_dim=x.shape[1], seed=seed, device=device)
        kmeans.train(aax.cpu().numpy())
        centers = torch.tensor(kmeans.get_centroids(), dtype=aax.dtype, device=aax.device)
        torch.save(centers, "{}/center_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b))

    if hasattr(data, 'drop_idx'):
        density[data.drop_idx] = 0

    # Obtain the centers
    indices = []
    # non_train = (train_mask  == 0)
    for center in centers:
        center = center.to(dtype=aax.dtype, device=aax.device)
        dist_map = torch.linalg.norm(aax - center, dim=1)
        dist_map = dist_map * density
        dist_map[indices] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
        # dist_map[non_train] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
        idx = int(torch.argmin(dist_map))
        indices.append(idx)

    return torch.tensor(indices)



def cluster_query(b, x, edge_index, train_mask, seed, device):
    seed_everything(seed)
    # Perform K-Means clustering:
    num_nodes = x.shape[0]
    aax = compute_norm_aax(x, edge_index, num_nodes)
    # import ipdb; ipdb.set_trace()
    if osp.exists("{}/center_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b)):
        centers = torch.load("{}/aax/center_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b), map_location='cpu')
    else:
        kmeans = Cluster(n_clusters=b, n_dim=x.shape[1], seed=seed, device=device)
        kmeans.train(aax.cpu().numpy())
        centers = torch.tensor(kmeans.get_centroids(), dtype=aax.dtype, device=aax.device)
        torch.save(centers, "{}/center_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b), map_location='cpu')

    # import ipdb; ipdb.set_trace()
    # Obtain the centers
    indices = []
    # non_train = (train_mask  == 0)
    for center in centers:
        center = center.to(dtype=aax.dtype, device=aax.device)
        dist_map = torch.linalg.norm(aax - center, dim=1)
        dist_map[indices] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
        # import ipdb; ipdb.set_trace()
        #
        #  dist_map[non_train] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
        idx = int(torch.argmin(dist_map))
        indices.append(idx)

    return torch.tensor(indices)


def split_cluster(b, partitions, num_parts, num_centers, seed, device, x_embed=None, method='default'):
    if method == 'inertia':
        part_size = []
        for i in range(num_parts):
            part_id = np.where(partitions == i)[0]
            x = x_embed[part_id]
            kmeans = Cluster(n_clusters=1, n_dim=x_embed.shape[1], seed=seed, device=device)
            kmeans.train(x.cpu())
            inertia = kmeans.get_inertia()
            part_size.append(inertia)

        part_size = np.rint(b * np.array(part_size) / sum(part_size)).astype(int)
        part_size = np.maximum(num_centers, part_size)
        i = 0
        while part_size.sum() - b != 0:
            if part_size.sum() - b > 0:
                i = num_parts - 1 if i <= 0 else i
                while part_size[i] <= 1:
                    i -= 1
                part_size[i] -= 1
                i -= 1
            else:
                i = 0 if i >= num_parts else i
                part_size[i] += 1
                i += 1

    elif method == 'size':
        part_size = []
        for i in range(num_parts):
            part_size.append(len(np.where(partitions == i)[0]))
        part_size = np.rint(b * np.array(part_size) / sum(part_size)).astype(int)
        part_size = np.maximum(num_centers, part_size)
        i = 0
        while part_size.sum() - b != 0:
            if part_size.sum() - b > 0:
                i = num_parts - 1 if i <= 0 else i
                while part_size[i] <= 1:
                    i -= 1
                part_size[i] -= 1
                i -= 1
            else:
                i = 0 if i >= num_parts else i
                part_size[i] += 1
                i += 1

    else:
        part_size = [b // num_parts for _ in range(num_parts)]
        for i in range(b % num_parts):
            part_size[i] += 1

    return part_size




def compute_norm_aax(x, edge_index, num_nodes):
    print("Start computing aax")
    # import ipdb; ipdb.set_trace()
    cache_path = osp.join(GLOBAL_RESULT_PATH, 'aax_{}.pt'.format(num_nodes))
    if os.path.exists(cache_path):
        res = torch.load(cache_path, map_location='cpu')
        return res
    else:
        new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
        adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                    sparse_sizes=(num_nodes, num_nodes))
        adj_matrix2 = adj.matmul(adj)
        aax = adj_matrix2.matmul(x)
        x = aax.to_dense()
        torch.save(x, cache_path)
        return x


def compute_density_aax(aax, num_nodes, b, device):
    print("Start computing density aax")
    if osp.exists("{}/density_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b)):
        aax_density = torch.load("{}/density_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b), map_location='cpu')
        return aax_density
    else:
        aax_kmeans = Cluster(n_clusters=b, n_dim=x.shape[1], seed=seed, device=device)
        aax_kmeans.train(aax)
        centers = aax_kmeans.get_centroids()
        label = aax_kmeans.predict(aax)

        aax = aax.to(device)
        centers = torch.tensor(centers[label], dtype=aax.dtype, device=aax.device)
        dist_map = torch.linalg.norm(aax - centers, dim=1).to(aax.dtype)
        aax_density = 1 / (1 + dist_map)
        torch.save(aax_density, "{}/density_aax_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, b))
        return aax_density



def gpart_query2(b, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part):
    seed_everything(seed)
    #data = cache_partitions(data)
    num_parts = int(np.ceil(b / num_centers))
    # compensation = 0
    if num_parts > max_part:
        num_parts = max_part
        compensation = compensation
    partitions = np.array(data.partitions[num_parts].cpu())
    num_nodes = x_embed.shape[0]
    edge_index = data.edge_index
    x = compute_norm_aax(x_embed, edge_index, num_nodes)
    # new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    # adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
    #                sparse_sizes=(num_nodes, num_nodes))
    # adj_matrix2 = adj.matmul(adj)
    # aax = adj_matrix2.matmul(x_embed)
    # x = aax.to_dense()
    # Get node representations
    # Determine the number of partitions and number of centers
    part_size = split_cluster(b, partitions, num_parts, num_centers, seed, device, x_embed=x)
    num_classes = data.y.max().item() + 1

    d_path = osp.join(GLOBAL_RESULT_PATH, 'density_x_{}_{}.pt'.format(num_nodes, num_classes))
    density = torch.load(d_path, map_location='cpu').to(device)
    # Iterate over each partition
    # indices = list(np.where(train_mask != 0)[0])
    # density_path = osp.join(GLOBAL_RESULT_PATH, 'density_aax_{}_{}.pt'.format(num_nodes, b))
    # aax_density = torch.load(density_path, map_location='cpu').to(device)
    
    
    indices = []
    for i in range(num_parts):
        part_id = np.where(partitions == i)[0]
        non_part_id = np.where(partitions != i)[0]
        masked_id = [i for i, x in enumerate(part_id) if x in indices]
        # masked_id = indices
        xi = x[part_id]

        n_clusters = part_size[i]
        if n_clusters <= 0:
            continue

        # Perform K-Means clustering:
        kmeans = Cluster(n_clusters=n_clusters, n_dim=xi.shape[1], seed=seed, device=device)
        kmeans.train(xi.cpu().numpy())
        centers = kmeans.get_centroids()

        # Compensating for the interference across partitions
        dist = None
        if compensation > 0:
            dist_to_center = torch.ones(x.shape[0], dtype=x.dtype, device=x.device) * np.infty
            for idx in indices:
                dist_to_center = torch.minimum(dist_to_center, torch.linalg.norm(x - x[idx], dim=1))
            dist = dist_to_center[part_id]

        # Obtain the centers
        for center in centers:
            center = torch.tensor(center, dtype=x.dtype, device=x.device)
            dist_map = torch.linalg.norm(xi - center, dim=1)
            N = xi.shape[0]
            percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
            if compensation > 0:
                dist_map -= dist * compensation
            # dist_map[non_part_id] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
            dist_map[masked_id] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
            sub_density = density[part_id]
            sub_density[masked_id] = 0
            # density[non_part_id] = 0
            id_sorted = dist_map.argsort(descending=True)
            dist_map[id_sorted] = percentile
            id_sorted = sub_density.argsort(descending=False)
            sub_density[id_sorted] = percentile
            score = dist_map + sub_density
            idx = int(torch.argmax(dist_map))
            masked_id.append(idx)
            indices.append(part_id[idx])

    return torch.tensor(indices)







def gpart_query(b, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part):
    seed_everything(seed)
    #data = cache_partitions(data)
    num_parts = int(np.ceil(b / num_centers))
    # compensation = 0
    if num_parts > max_part:
        num_parts = max_part
        compensation = compensation
    partitions = np.array(data.partitions[num_parts].cpu())
    num_nodes = x_embed.shape[0]
    edge_index = data.edge_index
    x = compute_norm_aax(x_embed, edge_index, num_nodes)
    # new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    # adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
    #                sparse_sizes=(num_nodes, num_nodes))
    # adj_matrix2 = adj.matmul(adj)
    # aax = adj_matrix2.matmul(x_embed)
    # x = aax.to_dense()
    # Get node representations
    # Determine the number of partitions and number of centers
    part_size = split_cluster(b, partitions, num_parts, num_centers, seed, device, x_embed=x)

    # Iterate over each partition
    # indices = list(np.where(train_mask != 0)[0])
    indices = []
    for i in range(num_parts):
        part_id = np.where(partitions == i)[0]
        masked_id = [i for i, x in enumerate(part_id) if x in indices]
        xi = x[part_id]

        n_clusters = part_size[i]
        if n_clusters <= 0:
            continue

        # Perform K-Means clustering:
        kmeans = Cluster(n_clusters=n_clusters, n_dim=xi.shape[1], seed=seed, device=device)
        kmeans.train(xi.cpu().numpy())
        centers = kmeans.get_centroids()

        # Compensating for the interference across partitions
        dist = None
        if compensation > 0:
            dist_to_center = torch.ones(x.shape[0], dtype=x.dtype, device=x.device) * np.infty
            for idx in indices:
                dist_to_center = torch.minimum(dist_to_center, torch.linalg.norm(x - x[idx], dim=1))
            dist = dist_to_center[part_id]

        # Obtain the centers
        for center in centers:
            center = torch.tensor(center, dtype=x.dtype, device=x.device)
            dist_map = torch.linalg.norm(xi - center, dim=1)
            if compensation > 0:
                dist_map -= dist * compensation
            dist_map[masked_id] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
            idx = int(torch.argmin(dist_map))
            masked_id.append(idx)
            indices.append(part_id[idx])

    return torch.tensor(indices)



def partition_hybrid(b, num_centers, data, train_mask, compensation, x_embed, seed, device, max_part):
    n_cluster = data.y.max().item() + 1
    _, reliability = density_query(b, x_embed, n_cluster, train_mask, seed, device)
    num_parts = int(np.ceil(b / num_centers))
    # compensation = 0
    if num_parts > max_part:
        num_parts = max_part
        compensation = compensation
    partitions = np.array(data.partitions[num_parts].cpu())
    num_nodes = x_embed.shape[0]
    edge_index = data.edge_index
    x = compute_norm_aax(x_embed, edge_index, num_nodes)
    part_size = split_cluster(b, partitions, num_parts, num_centers, seed, device, x_embed=x)

    # Iterate over each partition
    # indices = list(np.where(train_mask != 0)[0])
    indices = []
    for i in range(num_parts):
        part_id = np.where(partitions == i)[0]
        masked_id = [i for i, x in enumerate(part_id) if x in indices]
        xi = x[part_id]
        density_score = reliability[part_id]

        n_clusters = part_size[i]
        if n_clusters <= 0:
            continue

        # Perform K-Means clustering:
        kmeans = Cluster(n_clusters=n_clusters, n_dim=xi.shape[1], seed=seed, device=device)
        kmeans.train(xi.cpu().numpy())
        centers = kmeans.get_centroids()

        # Compensating for the interference across partitions
        dist = None
        if compensation > 0:
            dist_to_center = torch.ones(x.shape[0], dtype=x.dtype, device=x.device) * np.infty
            for idx in indices:
                dist_to_center = torch.minimum(dist_to_center, torch.linalg.norm(x - x[idx], dim=1))
            dist = dist_to_center[part_id]

        # Obtain the centers
        for center in centers:
            center = torch.tensor(center, dtype=x.dtype, device=x.device)
            ## diversity + representativeness

            dist_map = torch.linalg.norm(xi - center, dim=1)

            # dist_map *= density_score
            if compensation > 0:
                dist_map -= dist * compensation
            dist_map[masked_id] = torch.tensor(np.infty, dtype=dist_map.dtype, device=dist_map.device)
            idx = int(torch.argmin(dist_map))
            masked_id.append(idx)
            indices.append(part_id[idx])
    return indices



def compute_rw_norm_edge_index(edge_index, edge_weight = None, num_nodes = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                 device=edge_index.device)

    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight

    # L = I - A_norm.
    edge_index, tmp = add_self_loops(edge_index, edge_weight,
                                        fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp
    return edge_index, edge_weight



### utility functions for RIM (Nips 2021)

## rewrite the original implementation using sparse operations

def get_reliable_score(similarity, oracle_acc, num_class, prior = None):
    return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))

# def get_reliable_score2(similarity, norm_density, num_class):
#     return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))


def get_activated_node_dense(node,reliable_score,activated_node, adj_matrix2, th): 
    activated_vector=((adj_matrix2[node]*reliable_score)>th)+0
    activated_vector=activated_vector*activated_node
    # num_ones = torch.ones(adj_matrix2.shape[0])
    count = activated_vector.sum()
    return count.item(), activated_vector

def get_max_reliable_info_node_dense(high_score_nodes,activated_node,train_class,labels, oracle_acc, th, adj_matrix2, prior = None): 
    max_ral_node = 0
    max_activated_node = 0
    max_activated_num = 0 
    for node in high_score_nodes:
        if prior is not None:
            reliable_score = prior[node]
        else:
            reliable_score = oracle_acc
        activated_num,activated_node_tmp=get_activated_node_dense(node,reliable_score,activated_node, adj_matrix2, th)
        if activated_num > max_activated_num:
            max_activated_num = activated_num
            max_ral_node = node
            max_activated_node = activated_node_tmp        
    return max_ral_node,max_activated_node,max_activated_num

def update_reliability(idx_used,train_class,labels,num_node, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature, prior = None):
    activated_node = torch.zeros(num_node)
    num_class = labels.max().item()+1
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item()
        if prior is not None:
            prior_acc = prior[node]
        else:
            prior_acc = oracle_acc
        if node_label in train_class:
            total_score = 0.0
            for tmp_node in train_class[node_label]:
                total_score+=reliability_list[tmp_node]
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score(similarity_feature[node][tmp_node], oracle_acc, num_class)
            reliable_score = reliable_score/total_score
        else:
            reliable_score = oracle_acc
        reliability_list[node]=reliable_score
        activated_node+=((adj_matrix2[node]*reliable_score)>th)
    return torch.ones(num_node)-((activated_node>0).to(torch.int))

def compute_adj2(edge_index, num_nodes):
    print("Computing adj2")
    cache_path = osp.join(GLOBAL_RESULT_PATH, 'adj2_{}.pt'.format(num_nodes))
    if os.path.exists(cache_path):
        res = torch.load(cache_path, map_location='cpu')
        return res
    else:
        n_edge_index, n_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes = num_nodes)
        adj = SparseTensor(row=n_edge_index[0], col=n_edge_index[1], value=n_edge_weight,
                    sparse_sizes=(num_nodes, num_nodes))
        adj2 = adj.matmul(adj)
        torch.save(adj2, cache_path)
        return adj2


def compute_sim(norm_aax, num_nodes):
    print("compute sim")
    cache_path = osp.join(GLOBAL_RESULT_PATH, 'sim_{}.pt'.format(num_nodes))
    if os.path.exists(cache_path):
        res = torch.load(cache_path, map_location='cpu')
        return res
    else:
        similarity_feature = torch.mm(norm_aax, norm_aax.t())
        dis_range = torch.max(similarity_feature) - torch.min(similarity_feature)
        similarity_feature = (similarity_feature - torch.min(similarity_feature))/dis_range
        torch.save(similarity_feature, cache_path)
        return similarity_feature



# def large_scale_reliable_score_computation(reliability_list, train_class, node_label, oracle_acc, similarity_feature, num_class):
#     reliable_score = 0
#     same_label_nodes = torch.tensor(train_class[node_label])
#     reliability = torch.tensor(reliability_list)[same_label_nodes]
#     reliable_scores = 


def get_reliable_score_large(norm_features, node, tmp_node, oracle_acc, num_class):
    feature1 = norm_features[node]
    feature2 = norm_features[tmp_node]
    similarity = torch.dot(feature1, feature2)
    return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))



def large_scale_update_reliability(idx_used, train_class, labels, num_node, reliability_list, oracle_acc, th, adj_matrix2, norm_features):
    activated_node = torch.zeros(num_node)
    num_class = labels.max().item()+1
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item()
        if node_label in train_class:
            total_score = 0.0
            for tmp_node in train_class[node_label]:
                total_score+=reliability_list[tmp_node]
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score_large(norm_features, node, tmp_node, oracle_acc, num_class)
            reliable_score = reliable_score/total_score
        else:
            reliable_score = oracle_acc
        reliability_list[node]=reliable_score
        activated_node+=((adj_matrix2[node]*reliable_score)>th)
    return torch.ones(num_node)-((activated_node>0).to(torch.int))




def large_scale_get_active_node_dense(node,reliable_score,activated_node, adj_matrix2, th):
    adj_matrix2_row = adj_matrix2[node].to_dense()
    activated_vector=((adj_matrix2_row*reliable_score)>th)+0
    count = activated_vector.sum()
    return count.item(), activated_vector



def large_scale_get_max_reliable_info_node_dense(high_score_nodes,activated_node,train_class,labels, oracle_acc, th, adj_matrix2):
    max_ral_node = 0
    max_activated_node = 0
    max_activated_num = 0 
    for node in high_score_nodes:
        reliable_score = oracle_acc
        activated_num,activated_node_tmp=get_activated_node_dense(node,reliable_score,activated_node, adj_matrix2, th)
        if activated_num > max_activated_num:
            max_activated_num = activated_num
            max_ral_node = node
            max_activated_node = activated_node_tmp        
    return max_ral_node,max_activated_node,max_activated_num





def large_scale_rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, prior = None):
    seed_everything(seed)
    reliability_list = reliability_list[0]
    num_nodes = data.x.shape[0]
    features = data.x
    all_idx = torch.arange(num_nodes)

    adj_matrix2 = compute_adj2(edge_index, num_nodes)
    # norm_aax = compute_norm_aax(features, edge_index, num_nodes)
    labels = data.y
    idx_train = []
    idx_available = all_idx[train_mask].tolist()
    idx_available_temp = copy.deepcopy(idx_available)
    activated_node = torch.ones(num_nodes)
    count = 0
    train_class = {}
    while True:
        max_ral_node,max_activated_node,max_activated_num = large_scale_get_max_reliable_info_node_dense(idx_available_temp,activated_node,train_class,labels, oracle_acc, th, adj_matrix2) 
        idx_train.append(max_ral_node)
        idx_available.remove(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label]=list()
            train_class[node_label].append(max_ral_node)
        count += 1
        if count%batch_size == 0:
            activated_node = large_scale_update_reliability(idx_train,train_class,labels,num_nodes, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature)
        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)
        if count >= b or max_activated_num <= 0:
            break
    return torch.tensor(idx_train)
    


def lrim_wrapper(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, density_based = False):
    x_embed = data.x
    n_class = data.y.max().item() + 1
    if density_based:
        density = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, x_embed.shape[0], n_class))
        density = (density - density.min()) / (density.max() - density.min())
        return rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, prior = density)
    else:
        return rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed)


def rim_wrapper(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, density_based = False):
    x_embed = data.x
    n_class = data.y.max().item() + 1
    if density_based:
        density = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, x_embed.shape[0], n_class))
        density = (density - density.min()) / (density.max() - density.min())
        return rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, prior = density)
    else:
        return rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed)




def rim_query(b, edge_index, train_mask, data, oracle_acc, th, batch_size, reliability_list, seed, prior = None):
    seed_everything(seed)
    ## reliability list can be a prior
    reliability_list = reliability_list[0]
    num_nodes = data.x.shape[0]
    # similarity_feature = np.ones((num_node,num_node))
    features = data.x
    all_idx = torch.arange(num_nodes)

    adj_matrix2 = compute_adj2(edge_index, num_nodes).to_dense()
    norm_aax = compute_norm_aax(features, edge_index, num_nodes)
    similarity_feature = compute_sim(norm_aax, num_nodes)
    labels = data.y
    idx_train = []
    idx_available = all_idx[train_mask].tolist()
    idx_available_temp = copy.deepcopy(idx_available)
    # reliability_list = torch.ones(num_nodes)
    activated_node = torch.ones(num_nodes)
    count = 0
    train_class = {}
    while True:
        max_ral_node,max_activated_node,max_activated_num = get_max_reliable_info_node_dense(idx_available_temp,activated_node,train_class,labels, oracle_acc, th, adj_matrix2, prior = prior) 
        idx_train.append(max_ral_node)
        idx_available.remove(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label]=list()
            train_class[node_label].append(max_ral_node)
        count += 1
        if count%batch_size == 0:
            activated_node = update_reliability(idx_train,train_class,labels,num_nodes, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature, prior = prior)
        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)
        if count >= b or max_activated_num <= 0:
            break
    return torch.tensor(idx_train)



def label_prop(data, train_mask, edge_weight = None, num_layers = 3, alpha = 0.9):
    new_data = T.ToSparseTensor(remove_edge_index=False)(data)
    model = LabelPropagation(num_layers=3, alpha=0.9)
    out = model(data.y, data.adj_t, mask=train_mask)
    return out



def majority_counting(edge_index, pred):
    num_nodes = pred.shape[0]
    y_pred = torch.argmax(pred, dim=1)
    num_classes = y_pred.max().item() + 1
    label_counts = torch.zeros((num_nodes, num_classes), dtype=torch.int64)
    for begin, end in edge_index.t():
        label_counts[end, y_pred[begin]] += 1
    consistency_rate = label_counts[torch.arange(label_counts.shape[0]), y_pred] / label_counts.sum(dim=1)
    majority_result = torch.argmax(label_counts, dim=1)
    return majority_result, consistency_rate


def global_score_f(majority_result, gt, y_pred):
    majority_result = F.normalize(majority_result, p = 2, dim = -1)
    gt = F.normalize(gt, p = 2, dim = -1)
    score = torch.matmul(majority_result, gt.T)
    global_score = score[torch.arange(score.shape[0]), y_pred]
    ## inside every class, do a scaling
    for i in range(y_pred.max().item()+1):
        inside_class = torch.where(y_pred == i)[0]
        global_score[inside_class] = (global_score[inside_class] - torch.min(global_score[inside_class]))/(torch.max(global_score[inside_class]) - torch.min(global_score[inside_class]))
    return global_score



def active_llm_query(b, edge_index, x, confidence, seed, train_mask, data, reliability_list = None):
    idx_train = torch.arange(x.shape[0])[train_mask]
    knn_edge_index = knn_graph(
        x, 
        k = 6,
        batch = None,
        loop = False,
        flow = 'source_to_target',
        cosine=False,
        num_workers=1
    )

    knn_edge_index = to_undirected(knn_edge_index)

    num_nodes = x.shape[0]

    # A_struct = SparseTensor(row=edge_index[0], col=edge_index[1], value=None, sparse_sizes=(num_nodes, num_nodes))

    # A_knn = SparseTensor(row=knn_edge_index[0], col=knn_edge_index[1], value=None, sparse_sizes=(num_nodes, num_nodes))

    dim = x.shape[1]

    values = torch.ones(edge_index.shape[1])

    knn_values = torch.ones(knn_edge_index.shape[1])

    A_struct_X = spmm(edge_index, values, num_nodes, num_nodes, x)

    A_knn_X = spmm(knn_edge_index, knn_values, num_nodes, num_nodes, x)

    struct_norm_aax = F.normalize(A_struct_X, p = 2, dim = -1)
    struct_similarity_feature = torch.mm(struct_norm_aax, struct_norm_aax.t())
    struct_dis_range = torch.max(struct_similarity_feature) - torch.min(struct_similarity_feature)
    struct_similarity_feature = (struct_similarity_feature - torch.min(struct_similarity_feature))/struct_dis_range

    knn_norm_aax = F.normalize(A_knn_X, p = 2, dim = -1)
    knn_similarity_feature = torch.mm(knn_norm_aax, knn_norm_aax.t())
    knn_dis_range = torch.max(knn_similarity_feature) - torch.min(knn_similarity_feature)
    knn_similarity_feature = (knn_similarity_feature - torch.min(knn_similarity_feature))/knn_dis_range


    induced_labels = label_prop(data, train_mask)

    ## do neighbor voting 
    majority_result, consistency_rate = majority_counting(edge_index, induced_labels)
    
    y = torch.argmax(induced_labels, dim = 1)
    ## do global ranking
    num_classes = data.y.max().item() + 1
    one_hot = torch.zeros((y.shape[0], num_classes), dtype=torch.float32)
    one_hot[torch.arange(y.shape[0]), y] = 1
    global_score = global_score_f(induced_labels, one_hot, y)

    final_score = confidence * global_score * consistency_rate
    ## do 0-1 normalize
    norm_final_score = (final_score - torch.min(final_score))/(torch.max(final_score) - torch.min(final_score))

    if reliability_list is not None:
        reliability_list[-1] = norm_final_score
    return idx_train, norm_final_score


def train_lr(lr_model, x, y, epoch):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(lr_model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = lr_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return lr_model

def inference_lr(lr_model, x):
    with torch.no_grad():
        outputs = lr_model(x)
        return outputs


def progressive_llm_active_selection(b, data, confidence, train_mask, k = 4):
    ## total b budget
    ## first separate into k parts
    ## first randomly select b/k nodes from the pool
    data_idxs = torch.arange(data.x.shape[0])[train_mask]
    random_idxs = torch.randperm(data.x.shape[0])[:(b//k)]
    random_x = data.x[random_idxs]
    random_confidence = confidence[random_idxs]
    lr_model = LinearRegression(data.x.shape[1], 1)
    train_lr(lr_model, random_x, random_confidence, 100)
    lr_pred = inference_lr(lr_model, data.x)[train_mask]
    


def sole_confidence(b, confidence, train_mask):
    confidence[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(confidence, k=b)
    return indices






def hybrid_query(b, x_embed, edge_index, n_cluster, train_mask, confidence, seed, device):
    # Get propagated nodes
    # Perform K-Means as approximation
    num_nodes = x_embed.shape[0]
    new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                   sparse_sizes=(num_nodes, num_nodes))
    adj_matrix2 = adj.matmul(adj)
    aax = adj_matrix2.matmul(x_embed)
    km = kmedoids.KMedoids(5, method='fasterpam')
    c = km.fit(distmatrix)

    


def normalize_adj(edge_index, num_nodes, edge_weight = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def single_score(b, x, data, train_mask, seed, conf, device, pg_rank_score = 0.85):
    edges = [(int(i), int(j)) for i, j in zip(data.edge_index[0], data.edge_index[1])]
    nodes = [int(i) for i in range(x.shape[0])]
    data.g = nx.Graph()
    data.g.add_nodes_from(nodes)
    data.g.add_edges_from(edges)
    # Get centrality
    page = torch.tensor(list(pagerank(data.g, alpha = pg_rank_score).values()),
                        dtype=x.dtype, device=device)
    N = x.shape[0]
    kmeans = Cluster(n_clusters=data.y.max().item() + 1, n_dim=x.shape[1], seed=seed, device=device)
    kmeans.train(x)
    centers = kmeans.get_centroids()
    label = kmeans.predict(x)
    x = x.to(device)
    centers = torch.tensor(centers[label], dtype=x.dtype, device=x.device)
    dist_map = torch.linalg.norm(x - centers, dim=1).to(x.dtype)
    density = 1 / (1 + dist_map)

    aax = compute_norm_aax(x, data.edge_index, N)
    kmeans = Cluster(n_clusters=b, n_dim=x.shape[1], seed=seed, device=device)
    kmeans.train(x)
    centers = kmeans.get_centroids()
    label = kmeans.predict(x)
    x = x.to(device)
    centers = torch.tensor(centers[label], dtype=x.dtype, device=x.device)
    dist_map = torch.linalg.norm(x - centers, dim=1).to(x.dtype)
    aax_density = 1 / (1 + dist_map)


    # Get percentile
    percentile = (torch.arange(N, dtype=x.dtype, device=device) / N)
    id_sorted = density.argsort(descending=False)
    density[id_sorted] = percentile
    # id_sorted = entropy.argsort(descending=False)
    # entropy[id_sorted] = percentile
    id_sorted = page.argsort(descending=False)
    page[id_sorted] = percentile
    id_sorted = conf.argsort(descending=False)
    conf[id_sorted] = percentile
    id_sorted = aax_density.argsort(descending=False)
    aax_density[id_sorted] = percentile

    # Get score
    # alpha, beta, gamma = data.params['age']
    alpha, beta, gamma = 0.8, 0.1, 0.1
    age_score = alpha * page + beta * density + gamma * aax_density
    # age_score[np.where(train_mask == 0)[0]] = 0
    _, indices = torch.topk(age_score, k=b)
    return indices



def get_activated_node(node, activated_node, adj_matrix, th, oracle_acc):
    activated_vector=((adj_matrix[node] * oracle_acc)>th)+0
    activated_vector=activated_vector*activated_node
    count = activated_vector.sum()
    return count.item(), activated_vector

# def get_x_aax_dissimilarity(x_sim, train_idx, node):
#     x_sim_n = 1 / x_sim
#     # aax_sim_n = 1 / aax_sim
#     # x_sim_n[:, train_idx] = 0
#     # aax_sim_n[:, train_idx] = 0
#     # x_sim_n[train_idx, node] = 0
#     # aax_sim_n[train_idx, node] = 0
#     x_sim_n[node, train_idx] = 0
#     # aax_sim_n[node, train_idx] = 0
    
#     max_sim_x = torch.max(x_sim_n[node])
#     # max_sim_aax = torch.max(aax_sim_n[node])
    
#     return max_sim_x







def get_max_score_node_dense(high_score_nodes,activated_node,train_class,labels, th, adj_matrix2, oracle_acc, train_idx, density_score):
    ## dissimilarity score
    max_ral_node = 0
    max_activated_node = None
    max_activated_num = 0


    # activated_numbers = []
    activated_scores = torch.zeros(len(labels))
    sim_x_scores = torch.zeros(len(labels))
    sim_aax_scores = torch.zeros(len(labels))
    activated_nodes_total = [None for _ in range(len(labels))]
    if len(high_score_nodes) == 0:
        activated_scores = [0 for _ in range(len(labels))]
    else:
        for node in high_score_nodes:
            activated_num, activated_nodes = get_activated_node(node, activated_node, adj_matrix2, th, oracle_acc)
            # activated_numbers.append(activated_num)
            activated_scores[node] = activated_num
            activated_nodes_total[node] = activated_nodes
            ## consider density score as a prior of reliability
            ## consider X/AX dissimilarity 
            # max_sim_x, max_sim_aax = get_x_aax_dissimilarity(x_sim, aax_sim, train_idx, node)
            # sim_x_scores[node] = max_sim_x
            # sim_aax_scores[node] = max_sim_aax
    # if len(train_idx) > 0:
    #     sim_aax_scores = aax_sim[train_idx, :]
    #     #sim_aax_scores[train_idx, :] = 0
    #     sim_aax_scores, _ = sim_aax_scores.max(dim=0)
    #     sim_x_scores = x_sim[train_idx, :]
    #     #sim_x_scores[train_idx, :] = 0
    #     sim_x_scores, _ = sim_x_scores.max(dim=0)
    # else:
    #     sim_aax_scores = torch.zeros(len(labels))
    #     sim_x_scores = torch.zeros(len(labels))


    N = len(density_score)
    activated_scores = torch.tensor(activated_scores, dtype=density_score.dtype)
    id_sorted = activated_scores.argsort(descending=False)
    percentile = (torch.arange(N, dtype=density_score.dtype) / N)
    activated_order_scores = activated_scores.clone()
    activated_order_scores[id_sorted] = percentile
    
    id_sorted = density_score.argsort(descending=False)
    density_score[id_sorted] = percentile

    # sim_x_scores = sim_x_scores.to(density_score.dtype)
    # sim_aax_scores = sim_aax_scores.to(density_score.dtype)

    # id_sorted = sim_x_scores.argsort(descending=True)
    # sim_x_scores[id_sorted] = percentile

    # id_sorted = sim_aax_scores.argsort(descending=True)
    # sim_aax_scores[id_sorted] = percentile

    final_score = activated_order_scores + density_score
    final_score[train_idx] = 0
    max_score_node = torch.argmax(final_score)
    return max_score_node.item(), activated_nodes_total[max_score_node], activated_scores[max_score_node].item()


# def update_score(idx_train,train_class,labels,num_nodes, th, adj_matrix2):
#     activated_node = torch.zeros(num_nodes)
#     num_class = labels.max().item()+1
#     for node in idx_train:
#         activated_node+=((adj_matrix2[node])>th)
#     return torch.ones(num_nodes)-((activated_node>0).to(torch.int))



def update_score(idx_used,train_class,labels,num_node, reliability_list, oracle_acc, th, adj_matrix2, similarity_feature):
    activated_node = torch.zeros(num_node)
    num_class = labels.max().item()+1
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item()
        prior_acc = oracle_acc
        if node_label in train_class:
            total_score = 0.0
            for tmp_node in train_class[node_label]:
                total_score+=reliability_list[tmp_node]
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score(similarity_feature[node][tmp_node], oracle_acc, num_class)
            reliable_score = reliable_score/total_score
        else:
            reliable_score = oracle_acc
        reliability_list[node]=reliable_score
        activated_node+=((adj_matrix2[node]*reliable_score)>th)
    return torch.ones(num_node)-((activated_node>0).to(torch.int))




def iterative_score(b, edge_index, oracle_acc, train_mask, data, th, reliability_list, batch_size, seed, device):
    seed_everything(seed)
    num_nodes = data.x.shape[0]
    reliability_list = reliability_list[0]
    # similarity_feature = np.ones((num_node,num_node))
    features = data.x
    all_idx = torch.arange(num_nodes)

    # adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
    #                sparse_sizes=(num_nodes, num_nodes))
    n_clusters = data.y.max().item() + 1
    # _, density_score = density_query(b, data.x, n_clusters, train_mask, seed, device)
    density_score = torch.load("{}/density_x_{}_{}.pt".format(GLOBAL_RESULT_PATH, num_nodes, n_clusters))

    # # aax = adj.matmul(features).to_dense()
    # adj_matrix2 = adj.matmul(adj).to_dense()
    # aax = adj_matrix2.matmul(features)
    # norm_aax = F.normalize(aax, p = 2, dim = -1)
    # adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=None,
    #                 sparse_sizes=(num_nodes, num_nodes))

    # # # aax = adj.matmul(features).to_dense()
    # adj_matrix2 = adj.matmul(adj).to_dense()
    # norm_aax = compute_norm_aax(features, edge_index, num_nodes)
    # similarity_aax = torch.mm(norm_aax, norm_aax.t())
    # dis_range = torch.max(similarity_aax) - torch.min(similarity_aax)
    # similarity_aax = (similarity_aax - torch.min(similarity_aax))/dis_range

    # similarity_x = torch.mm(features, features.t())
    # dis_range = torch.max(similarity_x) - torch.min(similarity_x)
    # similarity_x = (similarity_x - torch.min(similarity_x))/dis_range
    adj_matrix2 = compute_adj2(edge_index, num_nodes).to_dense()
    norm_aax = compute_norm_aax(features, edge_index, num_nodes)
    similarity_feature = compute_sim(norm_aax, num_nodes)
    labels = data.y
    idx_train = []
    idx_available = all_idx.tolist()
    idx_available_temp = copy.deepcopy(idx_available)

    activated_node = torch.ones(num_nodes)
    count = 0
    train_class = {}

    # available_mask = torch.ones(num_nodes)
    # selected_nodes = []
    # update_period = 1
    while True:
        max_ral_node,max_activated_node,max_activated_num = get_max_score_node_dense(idx_available_temp,activated_node,train_class,labels, th, adj_matrix2, oracle_acc, idx_train, density_score) 
        idx_train.append(max_ral_node)
        idx_available.remove(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label]=list()
            train_class[node_label].append(max_ral_node)
        count += 1
        if count%batch_size == 0:
            activated_node = update_score(idx_train,train_class,labels,num_nodes, reliability_list,  oracle_acc, th, adj_matrix2, similarity_feature)
        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)
        if count >= b:
            break
    return torch.tensor(idx_train)



    
    






