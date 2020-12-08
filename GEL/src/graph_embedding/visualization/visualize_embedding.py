import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
import sys
sys.path.insert(0, './')
from graph_embedding.visualization import plot_util
from graph_embedding.visualization import graph_util


def plot_embedding2D(node_pos, inputFile, node_colors=None, labels=None):
    
    node_num = list(int(x) for x in node_pos.keys())
    # Convert list to an array 
    node_pos = list(node_pos.values()) 
    node_pos=np.array([np.array(xi) for xi in node_pos])
    print('shape',node_pos.shape)

    di_graph = graph_util.loadGraphFromEdgeListTxt(inputFile, directed=False)
    di_graph = di_graph.to_directed()
    _, embedding_dimension = node_pos.shape
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in node_num:
            pos[i] = node_pos[i, :]
        if node_colors is not None:
            nx.draw_networkx_nodes(di_graph, pos,
                                   node_color=node_colors,
                                   width=0.1, node_size=100,
                                   arrows=False, alpha=0.8,
                                   font_size=5, labels=labels)
        else:
            nx.draw_networkx(di_graph, pos, node_color=node_colors,
                             width=0.1, node_size=300, arrows=False,
                             alpha=0.8, font_size=12, labels=labels)


def expVis(X, res_pre, m_summ, node_labels=None, di_graph=None):
    print('\tGraph Visualization:')
    if node_labels:
        node_colors = plot_util.get_node_color(node_labels)
    else:
        node_colors = None
    plot_embedding2D(X, node_colors=node_colors,
                     di_graph=di_graph)
    plt.savefig('%s_%s_vis.pdf' % (res_pre, m_summ), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()
