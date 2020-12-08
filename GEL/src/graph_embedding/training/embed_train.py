
import ast
import logging
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from graph_embedding.NN.train_model import gae_model
from graph_embedding.MatrixFactorization import gf, grarep, hope, lap, line, node2vec, sdne
from graph_embedding.SVD.model import SVD_embedding
from graph_embedding.training.utils import *


def embedding_training(args, train_graph_filename):
   
    if args.method == 'GAE':
        g = read_for_gae(train_graph_filename)
    elif args.method == 'SVD':
        g = read_for_SVD(train_graph_filename, weighted=args.weighted)
    else:
        g = read_for_OpenNE(train_graph_filename, weighted=args.weighted)

    _embedding_training(args, G_=g)

    return


def _embedding_training(args, G_=None):
    seed=args.seed    
    if args.method == 'GAE':
        model = gae_model(args)
        G = G_[0]
        node_list = G_[1]
        model.train(G)
        # save embeddings
        model.save_embeddings(args.output, node_list)
    elif args.method == 'SVD':
        SVD_embedding(G_, args.output, size=args.dimensions)
    else:
        if args.method == 'Laplacian':
            model = lap.LaplacianEigenmaps(G_, rep_size=args.dimensions)

        elif args.method == 'GF':
            model = gf.GraphFactorization(G_, rep_size=args.dimensions,
                                          epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)

        elif args.method == 'HOPE':
            model = hope.HOPE(graph=G_, d=2, beta = args.beta)

        elif args.method == 'GraRep':
            model = grarep.GraRep(graph=G_, Kstep=args.kstep, dim=args.dimensions)

        elif args.method == 'DeepWalk':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, window=args.window_size, dw=True)

        elif args.method == 'node2vec':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)

        elif args.method == 'LINE':
            model = line.LINE(G_, epoch=args.epochs,
                              rep_size=args.dimensions, order=args.order)

        elif args.method == 'SDNE':
            encoder_layer_list = ast.literal_eval(args.encoder_list)
            model = sdne.SDNE(G_, encoder_layer_list=encoder_layer_list,
                              alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                              batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)
        else:
            raise ValueError(f'Invalid method: {args.method}')

        print("Saving embeddings...")
        model.save_embeddings(args.output)

    return
