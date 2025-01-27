import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

class LaplacianEigenmaps(object):
    def __init__(self, graph, rep_size=128):
        self.g = graph
        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = rep_size
        self.adj_mat = nx.to_numpy_array(self.g.G)
        self.vectors = {}
        self.embeddings = self.get_train()
        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def getAdj(self):
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = self.g.G[edge[0]][edge[1]]['weight']
        return adj

    def getLap(self):
       
        G = self.g.G.to_undirected()
        print('begin norm_lap_mat')
        norm_lap_mat = nx.normalized_laplacian_matrix(G)
        print('finish norm_lap_mat')
        return norm_lap_mat

    def get_train(self):
        lap_mat = self.getLap()
        print('finish getLap...')
        w, vec = eigsh(lap_mat, k=self.rep_size)
        print('finish eigh(lap_mat)...')
       
        return vec

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
