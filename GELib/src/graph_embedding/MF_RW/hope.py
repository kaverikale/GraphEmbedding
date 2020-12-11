
import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg


class HOPE(object):
    def __init__(self, graph, d, beta):
        '''
          d: embedding vector dimension
          beta: higher order coefficient
        '''
        self._d = d
        self._beta = beta
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()

    def learn_embedding(self):

        graph = self.g.G
        '''
        Returns the graph adjacency matrix as a NumPy matrix.
        '''
        A = nx.to_numpy_matrix(graph)

        '''
	Many high-order proximity measurements in graph can reflect the asymmetric transitivity. Moreover, we found that
	many of them share a general formulation which will facilitate the approximation of these proximities, that is:
	S = Inv(M_g) · (M_l)
	where M_g and M_l are both polynomial of matrices.
        '''

        #M_g = np.eye(graph.number_of_nodes())
        #M_l = np.dot(A, A)

        M_g = np.eye(len(graph.nodes)) - self._beta * A
        M_l = self._beta * A

        S = np.dot(np.linalg.inv(M_g), M_l)

        '''
	Singular Value Decomposition.

	Factorizes the matrix a into two unitary matrices U and Vh, and a 1-D array s of singular values (real, non-negative) such that a == U @ S @ Vt, where S is a suitably shaped matrix of zeros with main diagonal s.
        '''

        u, s, vt = lg.svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        
        self._X = np.concatenate((X1, X2), axis=1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :self._d // 2], self._X[j, self._d // 2:])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r

    @property
    def vectors(self):
        vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
