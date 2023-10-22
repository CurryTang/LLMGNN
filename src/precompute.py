import faiss 
from helper.data import get_dataset
from helper.active import compute_norm_aax
from helper.args import get_command_line_args
from helper.utils import load_yaml
import torch
import ipdb
import os.path as osp
from helper.active import GLOBAL_RESULT_PATH

datasets = ['cora']

## precompute the density_aax on gpu for all datasets

PATH = GLOBAL_RESULT_PATH

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


if __name__ == '__main__':
    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)
    # endpoint = params_dict.get('OPENAI_BASE', None)
    # if params_dict.get('OPENAI_KEY'):
    key = params_dict['OPENAI_KEY']
    data_path = params_dict['DATA_PATH']
    seeds = [0]
    for dataset in datasets:
        reliability_list = []
        data = get_dataset(seeds, dataset, args.split, args.data_format, params_dict['DATA_PATH'], None, 0, args.no_val, args.budget, 
                            args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, 1 - args.pl_noise, reliability_list, args.total_budget, 'none', False)
        x = data.x
        y = data.y
        cluster_1 = y.max().item() + 1
        cluster_2 = 70 
        cluster_3 = 20 * cluster_1
        cluster_4 = 40 * cluster_1

        cluster_5 = 560
        cluster_6 = 1120
        cluster_7 = 2240
        # import ipdb; ipdb.set_trace()
        n_dim = x.shape[1]
        cluster_model_1 = Cluster(n_clusters=cluster_1, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_2 = Cluster(n_clusters=cluster_2, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_3 = Cluster(n_clusters=cluster_3, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_4 = Cluster(n_clusters=cluster_4, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_5 = Cluster(n_clusters=cluster_5, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_6 = Cluster(n_clusters=cluster_6, n_dim=n_dim, seed=0, implementation='faiss')
        cluster_model_7 = Cluster(n_clusters=cluster_7, n_dim=n_dim, seed=0, implementation='faiss')

        num_nodes = x.shape[0]

        cluster_model_1.train(x)
        cluster_model_2.train(x)
        cluster_model_3.train(x)
        cluster_model_4.train(x)
        cluster_model_5.train(x)
        cluster_model_6.train(x)
        cluster_model_7.train(x)

        centers_1 = cluster_model_1.get_centroids()
        # import ipdb; ipdb.set_trace()
        centers_2 = cluster_model_2.get_centroids()
        centers_3 = cluster_model_3.get_centroids()
        centers_4 = cluster_model_4.get_centroids()
        centers_5 = cluster_model_5.get_centroids()
        centers_6 = cluster_model_6.get_centroids()
        centers_7 = cluster_model_7.get_centroids()
        centers_1 = torch.tensor(centers_1, dtype=x.dtype, device=x.device)
        centers_2 = torch.tensor(centers_2, dtype=x.dtype, device=x.device)
        centers_3 = torch.tensor(centers_3, dtype=x.dtype, device=x.device)
        centers_4 = torch.tensor(centers_4, dtype=x.dtype, device=x.device)
        centers_5 = torch.tensor(centers_5, dtype=x.dtype, device=x.device)
        centers_6 = torch.tensor(centers_6, dtype=x.dtype, device=x.device)
        centers_7 = torch.tensor(centers_7, dtype=x.dtype, device=x.device)
        torch.save(centers_1, osp.join(PATH, 'center_x_{}_{}.pt'.format(num_nodes, cluster_1)))
        torch.save(centers_2, osp.join(PATH, 'center_x_{}_{}.pt'.format(num_nodes, cluster_2)))
        torch.save(centers_3, osp.join(PATH, 'center_x_{}_{}.pt'.format(num_nodes, cluster_3)))
        torch.save(centers_4, osp.join(PATH, 'center_x_{}_{}.pt'.format(num_nodes, cluster_4)))

        ## save x density
        label1 = cluster_model_1.predict(x)
        label2 = cluster_model_2.predict(x)
        label3 = cluster_model_3.predict(x)
        label4 = cluster_model_4.predict(x)
        label5 = cluster_model_5.predict(x)
        label6 = cluster_model_6.predict(x)
        label7 = cluster_model_7.predict(x)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
        label3 = torch.tensor(label3)
        label4 = torch.tensor(label4)
        label5 = torch.tensor(label5)
        label6 = torch.tensor(label6)
        label7 = torch.tensor(label7)
        centers_1 = centers_1[label1]
        centers_2 = centers_2[label2]
        centers_3 = centers_3[label3]
        centers_4 = centers_4[label4]
        centers_5 = centers_5[label5]
        centers_6 = centers_6[label6]
        centers_7 = centers_7[label7]
        dist_map_1 = torch.linalg.norm(x - centers_1, dim=1)
        dist_map_2 = torch.linalg.norm(x - centers_2, dim=1)
        dist_map_3 = torch.linalg.norm(x - centers_3, dim=1)
        dist_map_4 = torch.linalg.norm(x - centers_4, dim=1)
        dist_map_5 = torch.linalg.norm(x - centers_5, dim=1)
        dist_map_6 = torch.linalg.norm(x - centers_6, dim=1)
        dist_map_7 = torch.linalg.norm(x - centers_7, dim=1)
        dist_map_1 = torch.tensor(dist_map_1, dtype=x.dtype, device=x.device)
        density_1 = 1 / (1 + dist_map_1)
        dist_map_2 = torch.tensor(dist_map_2, dtype=x.dtype, device=x.device)
        density_2 = 1 / (1 + dist_map_2)
        dist_map_3 = torch.tensor(dist_map_3, dtype=x.dtype, device=x.device)
        density_3 = 1 / (1 + dist_map_3)
        dist_map_4 = torch.tensor(dist_map_4, dtype=x.dtype, device=x.device)
        density_4 = 1 / (1 + dist_map_4)
        dist_map_5 = torch.tensor(dist_map_5, dtype=x.dtype, device=x.device)
        density_5 = 1 / (1 + dist_map_5)
        dist_map_6 = torch.tensor(dist_map_6, dtype=x.dtype, device=x.device)
        density_6 = 1 / (1 + dist_map_6)
        dist_map_7 = torch.tensor(dist_map_7, dtype=x.dtype, device=x.device)
        density_7 = 1 / (1 + dist_map_7)
        torch.save(density_1, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_1)))
        torch.save(density_2, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_2)))
        torch.save(density_3, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_3)))
        torch.save(density_4, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_4)))
        torch.save(density_5, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_5)))
        torch.save(density_6, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_6)))
        torch.save(density_7, osp.join(PATH, 'density_x_{}_{}.pt'.format(num_nodes, cluster_7)))


        aax = compute_norm_aax(x, data.edge_index, num_nodes)

        

        cluster_model_1.train(aax)
        cluster_model_2.train(aax)
        cluster_model_3.train(aax)
        cluster_model_4.train(aax)
        cluster_model_5.train(aax)
        cluster_model_6.train(aax)
        cluster_model_7.train(aax)

        ## save aax_center
        centers_1 = cluster_model_1.get_centroids()
        # import ipdb; ipdb.set_trace()
        centers_2 = cluster_model_2.get_centroids()
        centers_3 = cluster_model_3.get_centroids()
        centers_4 = cluster_model_4.get_centroids()
        centers_5 = cluster_model_5.get_centroids()
        centers_6 = cluster_model_6.get_centroids()
        centers_7 = cluster_model_7.get_centroids()
        centers_1 = torch.tensor(centers_1, dtype=x.dtype, device=x.device)
        centers_2 = torch.tensor(centers_2, dtype=x.dtype, device=x.device)
        centers_3 = torch.tensor(centers_3, dtype=x.dtype, device=x.device)
        centers_4 = torch.tensor(centers_4, dtype=x.dtype, device=x.device)
        centers_5 = torch.tensor(centers_5, dtype=x.dtype, device=x.device)
        centers_6 = torch.tensor(centers_6, dtype=x.dtype, device=x.device)
        centers_7 = torch.tensor(centers_7, dtype=x.dtype, device=x.device)
        torch.save(centers_1, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_1)))
        torch.save(centers_2, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_2)))
        torch.save(centers_3, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_3)))
        torch.save(centers_4, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_4)))
        torch.save(centers_5, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_5)))
        torch.save(centers_6, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_6)))
        torch.save(centers_7, osp.join(PATH, 'center_aax_{}_{}.pt'.format(num_nodes, cluster_7)))

        ## save aax density
        label1 = cluster_model_1.predict(aax)
        label2 = cluster_model_2.predict(aax)
        label3 = cluster_model_3.predict(aax)
        label4 = cluster_model_4.predict(aax)
        label5 = cluster_model_5.predict(aax)
        label6 = cluster_model_6.predict(aax)
        label7 = cluster_model_7.predict(aax)
        label1 = torch.tensor(label1)
        label2 = torch.tensor(label2)
        label3 = torch.tensor(label3)
        label4 = torch.tensor(label4)
        label5 = torch.tensor(label5)
        label6 = torch.tensor(label6)
        label7 = torch.tensor(label7)
        centers_1 = centers_1[label1]
        centers_2 = centers_2[label2]
        centers_3 = centers_3[label3]
        centers_4 = centers_4[label4]
        centers_5 = centers_5[label5]
        centers_6 = centers_6[label6]
        centers_7 = centers_7[label7]
        dist_map_1 = torch.linalg.norm(aax - centers_1, dim=1)
        dist_map_2 = torch.linalg.norm(aax - centers_2, dim=1)
        dist_map_3 = torch.linalg.norm(aax - centers_3, dim=1)
        dist_map_4 = torch.linalg.norm(aax - centers_4, dim=1)
        dist_map_5 = torch.linalg.norm(aax - centers_5, dim=1)
        dist_map_6 = torch.linalg.norm(aax - centers_6, dim=1)
        dist_map_7 = torch.linalg.norm(aax - centers_7, dim=1)
    
        dist_map_1 = torch.tensor(dist_map_1, dtype=x.dtype, device=x.device)
        density_1 = 1 / (1 + dist_map_1)
        dist_map_2 = torch.tensor(dist_map_2, dtype=x.dtype, device=x.device)
        density_2 = 1 / (1 + dist_map_2)
        dist_map_3 = torch.tensor(dist_map_3, dtype=x.dtype, device=x.device)
        density_3 = 1 / (1 + dist_map_3)
        dist_map_4 = torch.tensor(dist_map_4, dtype=x.dtype, device=x.device)
        density_4 = 1 / (1 + dist_map_4)
        dist_map_5 = torch.tensor(dist_map_5, dtype=x.dtype, device=x.device)
        density_5 = 1 / (1 + dist_map_5)
        dist_map_6 = torch.tensor(dist_map_6, dtype=x.dtype, device=x.device)
        density_6 = 1 / (1 + dist_map_6)
        dist_map_7 = torch.tensor(dist_map_7, dtype=x.dtype, device=x.device)
        density_7 = 1 / (1 + dist_map_7)
        torch.save(density_1, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_1)))
        torch.save(density_2, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_2)))
        torch.save(density_3, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_3)))
        torch.save(density_4, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_4)))
        torch.save(density_5, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_5)))
        torch.save(density_6, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_6)))
        torch.save(density_7, osp.join(PATH, 'density_aax_{}_{}.pt'.format(num_nodes, cluster_7)))
        # import ipdb; ipdb.set_trace()
