import datetime
import getpass
import json
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from graph_embedding.training.embed_train import embedding_training, load_embedding, read_node_labels, split_train_test_graph
from graph_embedding.evaluation.evaluation import LinkPrediction, NodeClassification
from .visualization.visualize_embedding import *


class Args():
    def __init__(self):
        #'Input graph file. Only accepted edgelist format.'
        self.dataset = ""
        #Output graph embedding file
        self.output = ""
        '''  choices=[
            'none',
            'link-prediction',
            'node-classification']
            'Choose to evaluate the embedding quality based on a specific prediction task. '
                             'None represents no evaluation, and only run for training embedding.'
        '''
        self.task = 'none'
    
        '''
            Testing set ratio for prediction tasks.'
                             'In link prediction, it splits all the known edges; '
                             'in node classification, it splits all the labeled nodes.'
        '''
        self.testingratio = 0.2
    
        '''
            'Number of random walks to start at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec'
        '''
        self.number_walks = 32
        '''
            'Length of the random walk started at each node. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec'
        '''
        self.walk_length = 64
        '''
            Number of parallel processes. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec
        ''' 
        self.workers = 8
    
        '''
            the dimensions of embedding for each node.
        '''
        self.dimensions = 100
        '''
            'Window size of word2vec model. '
                             'Only for random walk-based methods: DeepWalk, node2vec, struc2vec'
        '''
        self.window_size = 10

        '''
            'The training epochs of LINE, SDNE and GAE'
        '''
        self.epochs = 5

        '''
           p is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk explores.'
        '''
        self.p = 1.0
        '''
           q is a hyper-parameter for node2vec, '
                             'and it controls how fast the walk leaves the neighborhood of starting node.'
        '''
        self.q = 1.0
    
        '''
        The embedding learning method
            'Laplacian',
            'GF',
            'SVD',
            'HOPE',
            'GraRep',
            'node2vec',
            'DeepWalk'
            'LINE',
            'SDNE',
            'GAE'
        '''
        self.method = 'node2vec'
        '''
           'The label file for node classification'
        '''
        self.label_file = ''

        '''
           the negative ratio of LINE
        '''
        self.negative_ratio = 5

        '''
           Treat graph as weighted
        '''
        self.weighted = False
    
        '''
           Treat graph as directed
        '''
        self.directed = False

    
        '''
            Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order'
        '''
        self.order = 2
    
        '''
            coefficient for L2 regularization for Graph Factorization.
        '''
        self.weight_decay = 5e-4

        '''
            Use k-step transition probability matrix for GraRep.
        '''
        self.kstep = 4

        '''
            learning rate
        '''
        self.lr = 0.01
    
        '''
            alhpa is a hyperparameter in SDNE
        '''
        self.alpha = 0.3

        '''
            beta is a hyperparameter in SDNE
        '''
        self.beta = 0.01

        '''
            nu1 is a hyperparameter in SDNE
        '''
        self.nu1 = 1e-5
        '''
            nu2 is a hyperparameter in SDNE
        '''
        self.nu2 = 1e-4
        '''
            batch size of SDNE
        '''
        self.bs = 200
  
        '''
        a list of numbers of the neuron at each encoder layer, the last number is the
        imension of the output node representation
        
        '''
        self.encoder_list = '[1000, 128]'
   
        '''
           Dropout rate (1 - keep probability).
        '''
        self.dropout = 0
    
        '''
           Number of units in hidden layer.
        '''
        self.hidden = 32
    
        '''
            gae model selection: gcn_ae or gcn_vae
        '''
        self.gae_model_selection = 'gcn_ae'

        '''
            save evaluation performance
        '''
        self.eval_result_file = ''
   
        '''
           seed value
        '''
        self.seed = 0
    
    def set_values(self,dataset ,output , testingratio, number_walks, walk_length, workers, dimensions ,window_size, epochs, p,
                         q , method, label_file, negative_ratio, weighted , directed, order, weight_decay,   kstep,
                         lr , alpha, beta, nu1, nu2, bs, encoder_list, dropout, hidden, gae_model_selection, eval_result_file,
                         evaluation_metric , task , loss, seed):
        self.dataset = dataset
        self.output = output
        self.method = method
        self.testingratio = testingratio
        self.number_walks = number_walks
        self.walk_length = walk_length
        self.workers = workers
        self.dimensions = dimensions
        self.window_size = window_size
        self.epochs = epochs
        self.p = p
        self.q = q
        self.method = method
        self.label_file = label_file
        self.negative_ratio = negative_ratio
        self.weighted = weighted
        self.directed = directed
        self.order = order
        self.weight_decay = weight_decay
        self.kstep = kstep
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.bs = bs
        self.encoder_list = encoder_list
        self.dropout = dropout
        self.hidden = hidden
        self.gae_model_selection = gae_model_selection
        self.eval_result_file = eval_result_file
                  
        self.evaluation_metric = evaluation_metric
        self.task = task
        self.loss = loss                  
        self.seed = seed

def visualize_embedding(embedding_look_up, i, method):
    plot_embedding2D(embedding_look_up,i)
    line1, = plt.plot([1], label=method, linestyle='--')

    first_legend = plt.legend(handles=[line1], loc='upper right')
    ax = plt.gca().add_artist(first_legend)
    plt.show()

class Pipeline():
    
    def pipeline(self,
                  dataset = None,
                  output = None,
                  testingratio = 0.2,
                  number_walks = 32,
                  walk_length = 64,
                  workers = 8,
                  dimensions = 100,
                  window_size = 10,
                  epochs = 5,
                  p = 1.0,
                  q = 1.0,
                  method = 'node2vec',
                  label_file = '',
                  negative_ratio = 5,
                  weighted = False,
                  directed = False,
                  order = 2,
                  weight_decay = 5e-4,
                  kstep = 4,
                  lr = 0.01,
                  alpha = 0.3,
                  beta = 0.01,
                  nu1 = 1e-5,
                  nu2 = 1e-4,
                  bs = 200,
                  encoder_list = '[1000, 128]',
                  dropout = 0,
                  hidden = 32,
                  gae_model_selection = 'gcn_ae',
                  eval_result_file = '',
                  
                  evaluation_metric = None,
                  task = None,
                  loss = None,                  
                  seed = 0
                      ):
        args = Args()
        args.set_values(dataset ,output , testingratio, number_walks, walk_length, workers, dimensions ,window_size, epochs, p,
                         q , method, label_file, negative_ratio, weighted , directed, order, weight_decay,   kstep,
                         lr , alpha, beta, nu1, nu2, bs, encoder_list, dropout, hidden, gae_model_selection, eval_result_file,
                         evaluation_metric , task , loss, seed)
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        
        print('#' * 70)
        print('Embedding Method: %s, Evaluation Task: %s' % (args.method, args.task))
        print('#' * 70)
        result = None
        if args.task == 'link-prediction':
            G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(args.dataset, args.seed, weighted=args.weighted)
            time1 = time.time()
            embedding_training(args, train_graph_filename)
            embed_train_time = time.time() - time1
            print('Embedding Learning Time: %.2f s' % embed_train_time)
            embedding_look_up = load_embedding(args.output)
            time1 = time.time()
            print('Begin evaluation...')
            result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,args.seed)
            eval_time = time.time() - time1
            print('Prediction Task Time: %.2f s' % eval_time)
            os.remove(train_graph_filename)
            visualize_embedding(embedding_look_up, args.dataset, args.method)
        elif args.task == 'node-classification':
            if not label_file:
                raise ValueError("No input label file. Exit.")
            node_list, labels = read_node_labels(args.label_file)
            train_graph_filename = args.dataset
            time1 = time.time()
            embedding_training(args, train_graph_filename)
            embed_train_time = time.time() - time1
            print('Embedding Learning Time: %.2f s' % embed_train_time)
            embedding_look_up = load_embedding(args.output, node_list)
            time1 = time.time()
            print('Begin evaluation...')
            result = NodeClassification(embedding_look_up, node_list, labels, args.testingratio, args.seed)
            eval_time = time.time() - time1
            print('Prediction Task Time: %.2f s' % eval_time)
            visualize_embedding(embedding_look_up, args.dataset, args.method)
        else:
            train_graph_filename = args.dataset
            time1 = time.time()
            embedding_training(args, train_graph_filename)
            embedding_look_up = load_embedding(args.output)
            #print(embedding_look_up)
        
            embed_train_time = time.time() - time1
            print('Embedding Learning Time: %.2f s' % embed_train_time)
            visualize_embedding(embedding_look_up, args.dataset, args.method)

        if args.eval_result_file and result:
            _results = dict(
                input=args.dataset,
                task=args.task,
                method=args.method,
                dimension=args.dimensions,
                user=getpass.getuser(),
                date=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'),
                seed=args.seed,
            )

            if args.task == 'link-prediction':
                auc_roc, auc_pr, accuracy, f1 = result
                _results['results'] = dict(
                    auc_roc=auc_roc,
                    auc_pr=auc_pr,
                    accuracy=accuracy,
                    f1=f1,
                )
            else:
                accuracy, f1_micro, f1_macro = result
                _results['results'] = dict(
                    accuracy=accuracy,
                    f1_micro=f1_micro,
                    f1_macro=f1_macro,
                )

            with open(args.eval_result_file, 'a+') as wf:
                print(json.dumps(_results, sort_keys=True), file=wf)
             

