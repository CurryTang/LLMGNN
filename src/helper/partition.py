import numpy as np

from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from numpy import linalg as lin

import json
import matplotlib
import matplotlib.pyplot as plt

import torch

from networkx.algorithms.community.quality import modularity
from networkx.algorithms.boundary import edge_boundary
from networkx.algorithms.components import number_connected_components
from networkx.utils.mapped_queue import MappedQueue

font = {'family': 'normal',
        'weight': 'normal',
        'size': 32}
matplotlib.rc('font', **font)


class GraphPartition:

    def __init__(self, graph, x, num_classes):

        self.graph = graph
        self.x = x
        self.n_cluster = num_classes
        self.costs = []

    def clauset_newman_moore(self, num_part=-1, weight=None, q_break=0):
        """
        Find communities in graph using Clauset-Newman-Moore greedy modularity maximization.

        Greedy modularity maximization begins with each node in its own community
        and joins the pair of communities that most increases (least decrease) modularity
        until q_break.

        Modified from
        https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities
        """

        # Count nodes and edges
        N = len(self.graph.nodes())
        m = sum([d.get("weight", 1) for u, v, d in self.graph.edges(data=True)])
        q0 = 1.0 / (2.0 * m)

        # Map node labels to contiguous integers
        label_for_node = {i: v for i, v in enumerate(self.graph.nodes())}
        node_for_label = {label_for_node[i]: i for i in range(N)}

        # Calculate edge weight
        if weight is not None:
            edge_weight = []
            for edge in self.graph.edges:
                edge_weight.append(torch.linalg.norm(self.x[edge[0]] - self.x[edge[1]]).item())
            edge_weight = torch.tensor(edge_weight)
            edge_weight -= edge_weight.min()
            edge_weight /= edge_weight.max()
            attrs = {}
            for edge, distance in zip(self.graph.edges, list(edge_weight)):
                attrs[edge] = {'distance': distance}
            weight = 'distance'

        # Calculate degrees
        k_for_label = self.graph.degree(self.graph.nodes(), weight=weight)
        k = [k_for_label[label_for_node[i]] for i in range(N)]

        # Initialize community and merge lists
        communities = {i: frozenset([i]) for i in range(N)}

        # Initial modularity and homophily
        partition = [[label_for_node[x] for x in c] for c in communities.values()]
        q_cnm = modularity(self.graph, partition)

        # Initialize data structures
        # CNM Eq 8-9 (Eq 8 was missing a factor of 2 (from A_ij + A_ji)
        # a[i]: fraction of edges within community i
        # dq_dict[i][j]: dQ for merging community i, j
        # dq_heap[i][n] : (-dq, i, j) for communitiy i nth largest dQ
        # H[n]: (-dq, i, j) for community with nth largest max_j(dQ_ij)
        a = [k[i] * q0 for i in range(N)]
        dq_dict = {
            i: {
                j: 2 * q0 - 2 * k[i] * k[j] * q0 * q0
                for j in [node_for_label[u] for u in self.graph.neighbors(label_for_node[i])]
                if j != i
            }
            for i in range(N)
        }
        dq_heap = [
            MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
        ]
        H = MappedQueue([dq_heap[i].heap[0] for i in range(N) if len(dq_heap[i]) > 0])

        # Merge communities until we can't improve modularity
        while len(H) > 1:
            # Find best merge
            # Remove from heap of row maxes
            # Ties will be broken by choosing the pair with lowest min community id
            try:
                dq, i, j = H.pop()
            except IndexError:
                break
            dq = -dq

            # Remove best merge from row i heap
            dq_heap[i].pop()

            # Push new row max onto H
            if len(dq_heap[i]) > 0:
                H.push(dq_heap[i].heap[0])

            # If this element was also at the root of row j, we need to remove the
            # duplicate entry from H
            if dq_heap[j].heap[0] == (-dq, j, i):
                H.remove((-dq, j, i))
                # Remove best merge from row j heap
                dq_heap[j].remove((-dq, j, i))
                # Push new row max onto H
                if len(dq_heap[j]) > 0:
                    H.push(dq_heap[j].heap[0])
            else:
                # Duplicate wasn't in H, just remove from row j heap
                dq_heap[j].remove((-dq, j, i))

            # Stop when change is non-positive 0
            if 0 < num_part == len(communities):
                break
            elif dq <= q_break:
                break

            # New modularity and homophily
            q_cnm += dq

            # Perform merge
            communities[j] = frozenset(communities[i] | communities[j])
            del communities[i]

            # Get list of communities connected to merged communities
            i_set = set(dq_dict[i].keys())
            j_set = set(dq_dict[j].keys())
            all_set = (i_set | j_set) - {i, j}
            both_set = i_set & j_set

            # Merge i into j and update dQ
            for k in all_set:

                # Calculate new dq value
                if k in both_set:
                    dq_jk = dq_dict[j][k] + dq_dict[i][k]
                elif k in j_set:
                    dq_jk = dq_dict[j][k] - 2.0 * a[i] * a[k]
                else:
                    # k in i_set
                    dq_jk = dq_dict[i][k] - 2.0 * a[j] * a[k]

                # Update rows j and k
                for row, col in [(j, k), (k, j)]:
                    # Save old value for finding heap index
                    if k in j_set:
                        d_old = (-dq_dict[row][col], row, col)
                    else:
                        d_old = None
                    # Update dict for j,k only (i is removed below)
                    dq_dict[row][col] = dq_jk
                    # Save old max of per-row heap
                    if len(dq_heap[row]) > 0:
                        d_oldmax = dq_heap[row].heap[0]
                    else:
                        d_oldmax = None
                    # Add/update heaps
                    d = (-dq_jk, row, col)
                    if d_old is None:
                        # We're creating a new nonzero element, add to heap
                        dq_heap[row].push(d)
                    else:
                        # Update existing element in per-row heap
                        dq_heap[row].update(d_old, d)
                    # Update heap of row maxes if necessary
                    if d_oldmax is None:
                        # No entries previously in this row, push new max
                        H.push(d)
                    else:
                        # We've updated an entry in this row, has the max changed?
                        if dq_heap[row].heap[0] != d_oldmax:
                            H.update(d_oldmax, dq_heap[row].heap[0])

            # Remove row/col i from matrix
            i_neighbors = dq_dict[i].keys()
            for k in i_neighbors:
                # Remove from dict
                dq_old = dq_dict[k][i]
                del dq_dict[k][i]
                # Remove from heaps if we haven't already
                if k != j:
                    # Remove both row and column
                    for row, col in [(k, i), (i, k)]:
                        # Check if replaced dq is row max
                        d_old = (-dq_old, row, col)
                        if dq_heap[row].heap[0] == d_old:
                            # Update per-row heap and heap of row maxes
                            dq_heap[row].remove(d_old)
                            H.remove(d_old)
                            # Update row max
                            if len(dq_heap[row]) > 0:
                                H.push(dq_heap[row].heap[0])
                        else:
                            # Only update per-row heap
                            dq_heap[row].remove(d_old)

            del dq_dict[i]
            # Mark row i as deleted, but keep placeholder
            dq_heap[i] = MappedQueue()
            # Merge i into j and update a
            a[j] += a[i]
            a[i] = 0

        communities = [
            [label_for_node[i] for i in c] for c in communities.values()
        ]
        return sorted(communities, key=len, reverse=True)

    def agglomerative_clustering(self, communities, min_clusters=2):
        """
        Agglomerative Clustering: Ward's Linkage Method
        """

        n_clusters = list(range(min_clusters, len(communities)))
        n_clusters.reverse()
        partitions = {}

        dist, x_com = self.community_linkage(communities, full=True)

        num_clusters = len(communities)
        while num_clusters > min(n_clusters):

            sorted_communities = sorted(communities, key=lambda c: len(c), reverse=True)
            partitions[num_clusters] = torch.zeros(self.x.shape[0], dtype=torch.int)
            for i, com in enumerate(sorted_communities):
                partitions[num_clusters][com] = i

            merge_cost, closest_idx = torch.min(dist, dim=1)
            j = torch.argmin(merge_cost).item()
            i = closest_idx[j].item()
            assert i > j

            communities[j].extend(communities[i])
            del communities[i]
            x_com = torch.cat((x_com[0:i], x_com[i + 1:]), dim=0)
            x_com[j] = self.x[communities[j]].mean(axis=0)

            dist = torch.cat((dist[0:i], dist[i + 1:]), dim=0)
            dist = torch.cat((dist[:, 0:i], dist[:, i + 1:]), dim=1)
            num_clusters -= 1

            for k in range(len(communities)):
                if k == j:
                    continue
                nk, nj = len(communities[k]), len(communities[j])
                n = nk * nj / (nk + nj)
                d = torch.linalg.norm(x_com[j] - x_com[k])
                dist[k, j] = d * n
                dist[j, k] = d * n

            cost = merge_cost.min().item()
            self.costs.append(cost)

        return partitions

    def community_linkage(self, communities, full=True):

        n = self.x.shape[1]
        x_com = []
        for com in communities:
            x_com.append(self.x[com].mean(axis=0))
        x_com = torch.stack(x_com, dim=0)

        linkage = torch.linalg.norm(
            x_com.reshape(1, -1, n) - x_com.reshape(-1, 1, n), dim=2
        )
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                ni, nj = len(communities[i]), len(communities[j])
                n = ni * nj / (ni + nj)
                linkage[i, j] *= n
                linkage[j, i] *= n
        linkage += torch.diag(torch.ones(linkage.shape[0]) * float("Inf"))

        if full:
            return linkage, x_com
        return linkage
