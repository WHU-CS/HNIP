from collections import defaultdict, Iterable
import random
from itertools import product, permutations
# import time
from six import iterkeys
import math
from gensim.models import Word2Vec
from tqdm import tqdm


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def get_sum_weight_for_node(self, nodeId, T, lamb):
        weight_list = []
        # sum_weight = 0
        for edge in self[nodeId]:
            weight_list.append(edge.get_cal_weight(T, lamb))
            # sum_weight += edge.get_cal_weight(T, lamb)
        # return sum_weight
        return weight_list

    def get_neighbors_by_node(self, nodeId):
        neighbors = []

        for edge in self[nodeId]:
            neighbors.append(edge.get_node())
        return neighbors

    def get_probability(self, nodeId, T, lamb):
        proba = []
        weight_list = self.get_sum_weight_for_node(nodeId,T,lamb)
        sum_pro = 0
        for i in range(len(weight_list)):
            sum_pro += weight_list[i]
        for i in range(len(weight_list)):
            proba.append(weight_list[i]/sum_pro)
        # for edge in self[nodeId]:
        #     # sum_pro = self.get_sum_weight_for_node(nodeId, T, lamb)
        #     # proba.append(edge.get_cal_weight(T, lamb) / sum_pro)
        #     proba.append()
        return proba

    def get_time_by_node(self, nodeId):
        times = []
        for edge in self[nodeId]:
            times.append(edge.get_time())
        return times

class edge_list:
    def __init__(self, G, T, lamb):
        self.edges = {}
        self.pros = {}
        self.times = {}
        self.T = T
        self.lamb = lamb

        for v in tqdm(G.keys()):
            self.edges[v] = G.get_neighbors_by_node(v)
            self.pros[v] = G.get_probability(v, T, lamb)
            self.times[v] = G.get_time_by_node(v)

    def get_neighbors_by_node(self, nodeId):
        return self.edges[nodeId]

    def get_pro_by_node(self, nodeId):
        return self.pros[nodeId]

    def get_time_by_node(self, nodeId):
        return self.times[nodeId]


class Edge(object):
    def __init__(self, node, weight, times):
        self.node = node
        self.weight = weight
        self.times = times

    def get_node(self):
        return self.node

    def get_weight(self):
        return self.weight

    def get_time(self):
        return self.times

    def get_cal_weight(self, T, lamb):
        return math.exp(-lamb*(T - self.times)) * self.weight


def from_adjlist(file_path):
    G = Graph()
    with open(file_path) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(':')
            edges = lineArr[1].split(';')
            edges = edges[:-1]
            for e in edges:
                item = e.split(',')
                item = item[:-1]
                edge = Edge(int(item[0]), float(item[1]), float(item[2]))
                G[int(lineArr[0])].append(edge)
    return G


def get_random_node(node_list, pro_list):
    x = random.uniform(0, 1)
    index = 0
    cum_pro = 0.0

    # print(len(node_list))
    for item, item_pro in zip(node_list, pro_list):
        cum_pro += item_pro

        if x < cum_pro: break
        index += 1
    return item, index


def bulid_temporal_walk_corpus(G, edgelist, num_path, path_length, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())

    for cnt in range(num_path):
        rand.shuffle(nodes)
        for node in tqdm(nodes):
            path = [node]
            cur_time = 0
            for tag in range(path_length):
            # while len(path) < path_length:
                cur = path[-1]
                if len(G[cur]) > 0:
                    # print(cur)
                    time_list = edgelist.get_time_by_node(cur)
                    # print(len(time_list))
                    node, index = get_random_node(edgelist.get_neighbors_by_node(cur), edgelist.get_pro_by_node(cur))
                    if time_list[index] > cur_time:
                        cur_time = time_list[index]
                        path.append(node)
                # else:
                #     break
            walks.append([str(node) for node in path])
    return walks


if __name__ == '__main__':


    print('load data')
    G = from_adjlist(input_file)

    print('get_edges')

    edges = edge_list(G, T=1, lamb=1)

    print('Walking...')

    walks = bulid_temporal_walk_corpus(G, edges, 5, 20)

    size = 0

    for path in walks:
        size += len(path)
    print(size)

    model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, hs=1, workers=1)

    model.wv.save_word2vec_format(output_file)