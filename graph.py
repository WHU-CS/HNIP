import scipy.io as sio
import random
from scipy.sparse import dok_matrix
import numpy as np
from collections import defaultdict
import math
import random
import gc
import sys
import time
class Dotdict(dict):

    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Graph(object):
    def __init__(self, file_path, ng_sample_ratio, T, tao, walk_times, walk_length, restart_ratio):
        self.st = 0
        if (ng_sample_ratio > 0):
            self.__negativeSample(int(ng_sample_ratio * self.E))
        self.is_epoch_end = False
        self.adj_matrix, self.N = self.__get_adj_matrix(file_path, T, tao)
        # print(self.N)
        self.E = self.adj_matrix.count_nonzero()/2
        print('number of edges:' + str(self.E))
        self.order = np.arange(self.N)
        self.walks = self.__get_walks(file_path, T, tao, walk_times, walk_length, restart_ratio)

    def __get_adj_matrix(self, file_path, T, alpha):
        print('getting adJ...')
        time0 =time.time()
        ad_list = defaultdict(list)
        N = 0
        with open(file_path) as fr:
            for line in fr.readlines():
                line = line.strip().split(':')
                edges = line[1].strip().split(";")
                edges = edges[:-1]
                node_s = int(line[0])
                for edge in edges:
                    item = edge.split(',')
                    item = item[:-1]
                    node_t = int(item[0])
                    weight = float(item[1])
                    time_ = float(item[2])
                    e = Edge_(node_t, weight, time_)
                    ad_list[node_s].append(e)
        ad_mat = dok_matrix((len(ad_list), len(ad_list)), dtype=np.float)
        N = len(ad_list)

        # avg_degree = 0

        print('number of nodes = ' + str(N))

        for node_id in ad_list.keys():  # 时间复杂度为O(|v|)
            # avg_degree += len(ad_list[node_id])
            for edge in ad_list[node_id]:
                i = node_id
                j = edge.get_node()
                weight = edge.get_weight()
                time_ = edge.get_time()
                if i != j:
                    if i >= N:
                        print('i = ' + str(i))
                    if j >= N :
                        print('j = ' + str(j))
                    ad_mat[i, j] += weight * math.exp(alpha * -(T - time_))

        del ad_list
        gc.collect()
        time1 = time.time()
        print('finish ad_matrix construction, run time :' + str(time1 - time0))

        return ad_mat.tocsr(), N

    def __get_walks(self, file_path, T, alpha, num_paths, path_length, restart_ratio, rand=random.Random(0)):
        print('walking...')
        time0 = time.time()
        G = from_adjlist(file_path)
        edges = edgelist(G, T, alpha)

        walks = []
        nodes = list(G.nodes())

        for cnt in range(num_paths):
            rand.shuffle(nodes)
            for node in nodes:
                path = [node]
                while len(path) < path_length:
                    cur = path[-1]
                    if (len(G[cur])) > 0:
                        if rand.random() >= restart_ratio:
                            path.append(get_random_node(edges.get_edges_by_node(cur), edges.get_pro_by_node(cur)))
                        else:
                            path.append(path[0])
                    else:
                        break
                walks.append([int(node) for node in path])
        del G
        gc.collect()
        time1 = time.time()
        print('finish random walk, run time :' + str(time1 - time0))
        return walks

    def __negativeSample(self, ngSample):
        print("negative Sampling")
        size = 0
        while(size < ngSample):
            xx = random.randint(0, self.N-1)
            yy = random.randint(0, self.N-1)
            if (xx == yy or self.adj_matrix[xx, yy] != 0):
                continue
            self.adj_matrix[xx, yy] = -1
            self.adj_matrix[yy, xx] = -1
            size += 1
        print("negative sampling done")

    def sample(self, batch_size, do_shuffle = True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.order[0: self.N])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False
        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.order[self.st:en]
        mini_batch.X = self.adj_matrix[index].toarray()
        mini_batch.adjacent_matrix = self.adj_matrix[index].toarray()[:][:,index]
        if en == self.N:
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch

    def get_mini_batch_by_walk(self, walk):
        mini_batch = Dotdict()
        index = walk
        mini_batch.X = self.adj_matrix[index].toarray()
        '''
        just preserve the similarity between adjacency node in the walk path
        '''
        mini_batch.adjacent_matrix = np.zeros((len(index), len(index)))
        for i in range(len(index)):
            if i+1 < len(index):

                mini_batch.adjacent_matrix[i][i+1] = 1
                mini_batch.adjacent_matrix[i+1][i] = 1

        '''
        preserve the similarity between every node pair in the walk path
        '''
        return mini_batch

class Edge_(object):
    def __init__(self, node, weight, time):
        self.node = node
        self.weight = weight
        self.time = time

    def get_node(self):
        return self.node

    def get_weight(self):
        return self.weight

    def get_time(self):
        return self.time


class G(defaultdict):
    def __init__(self):
        super(G, self).__init__(list)

    def nodes(self):
        return self.keys()

    def remove_self_loops(self):
        removed = 0
        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        return self

    def get_sum_weight_for_node(self, nodeId, T, lamb):
        weight_list = []

        # sum_weight = 0
        for edge in self[nodeId]:
            weight_list.append(edge.get_cal_weight(T, lamb))
            # sum_weight += edge.get_cal_weight(T, lamb)
        return weight_list

    def get_edges_by_node(self, nodeId):
        edges = []
        for edge in self[nodeId]:
            edges.append(edge.get_node())
        return edges

    def get_probability(self, nodeId, T, lamb):
        probabilities = []
        weight_list = self.get_sum_weight_for_node(nodeId, T, lamb)
        sum_pro = 0
        # for edge in self[nodeId]:
        #         #     sum_pro = self.get_sum_weight_for_node(nodeId, T, lamb)
        #         #     probabilities.append(edge.get_cal_weight(T, lamb) / sum_pro)
        for i in range(len(weight_list)):
            sum_pro += weight_list[i]
        for i in range(len(weight_list)):
            probabilities.append(weight_list[i] / sum_pro)
        return probabilities


class edgelist:
    def __init__(self, G, T, lamb):
        self.edges = {}
        self.pros = {}
        self.T = T
        self.lamb = lamb
        for v in G.keys():
            self.edges[v] = G.get_edges_by_node(v)
            self.pros[v] = G.get_probability(v, T, lamb)

    def get_edges_by_node(self, nodeId):
        return self.edges[nodeId]

    def get_pro_by_node(self, nodeId):
        return self.pros[nodeId]


def from_adjlist(file_):
    g = G()
    with open(file_) as f:
        for line in f:
            lineArr = line.strip().split(":")
            edges = lineArr[1].split(';')
            edges = edges[:-1]
            for e in edges:
                item = e.split(',')
                item = item[:-1]
                edge = Edge(int(item[0]), float(item[1]), float(item[2]))
                g[int(lineArr[0])].append(edge)
    return g


class Edge(object):
    def __init__(self, node, weight, time):
        self.node = node
        self.weight = weight
        self.time = time

    def get_node(self):
        return self.node

    def get_weight(self):
        return self.weight

    def get_time(self):
        return self.time

    def get_cal_weight(self, T, lamb):
        return math.exp(-lamb*(T - self.time)) * self.weight


def get_random_node(node_list, pro_list):
    x = random.uniform(0, 1)
    cum_pro = 0.0
    for item, item_pro in zip(node_list, pro_list):
        cum_pro += item_pro
        if x < cum_pro: break

    return item