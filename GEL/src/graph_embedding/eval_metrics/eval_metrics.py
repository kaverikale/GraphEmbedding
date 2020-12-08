import numpy as np


precision_pos = [2, 10, 100, 200, 300, 500, 1000]

def computePrecisionCurveAndRank(text_edges, true_digraph, max_k=-1):
    if max_k == -1:
        max_k = len(text_edges)
    else:
        max_k = min(max_k, len(text_edges))

    #sorted_edges = sorted(text_edges, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(text_edges[i][0], text_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors

def computeMRR(text_edges, y_test, y_pred, original_graph, max_k=-1):
    node_num = len(original_graph.nodes)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    i=0
    for (st, ed) in text_edges:
        
        if y_pred[i] == 1: 
            node_edges[int(st)].append((st, ed, 1.0))
            node_edges[int(ed)].append((ed, st, 1.0))
        i = i+1
    node_RR = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if len(node_edges[i]) != 0:
            count += 1
        
        _, delta_factors = computePrecisionCurveAndRank(node_edges[i], original_graph, max_k)
        
        if(delta_factors.count(1.0) > 0):
            index = delta_factors.index(1.0)
            node_RR[i] = 1.0/(index + 1.0)
        else:
            node_RR[i] = 0
    return sum(node_RR) / count
def calculateMAP(text_edges, y_test, y_pred, original_graph, max_k=-1):
    node_num = len(original_graph.nodes)
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    i=0
    for (st, ed) in text_edges:
        
        if y_pred[i] == 1: 
            node_edges[int(st)].append((st, ed, 1.0))
            node_edges[int(ed)].append((ed, st, 1.0))
        i = i+1

    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if len(node_edges[i]) != 0:
            count += 1
        print(i, node_edges[i])
        precision_scores, delta_factors = computePrecisionCurveAndRank(node_edges[i], original_graph, max_k)
        precision_rectified = [p * d for p,d in zip(precision_scores,delta_factors)]
        if(sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count

def computeMacroF1(predicted_nodes, true_nodes):
    macro = f1_score(true_nodes, predicted_nodes, average='macro') 
    return macro 

def computeMicroF1(predicted_nodes, true_nodes):
    micro = f1_score(true_nodes, predicted_nodes, average='micro')
    return micro


    



