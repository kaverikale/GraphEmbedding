
import datetime
import getpass
import json
import os
import random
import time

from .pipeline import Pipeline

def more_main():
    pipeline = Pipeline()
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'Laplacian', task = 'link-prediction', dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'GF', task = 'link-prediction',dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'SVD', task = 'link-prediction',dimensions = 10)
    pipeline.pipeline(dataset = '../../data/facebook.txt', output = '../../embeddings/emb.txt', method = 'HOPE', task = 'link-prediction',dimensions = 2)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'GraRep', task = 'link-prediction',dimensions = 16)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'node2vec', task = 'link-prediction',dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'DeepWalk', task = 'link-prediction',dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'LINE', task = 'link-prediction',dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'SDNE', task = 'link-prediction',dimensions = 10)
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'GAE', task = 'link-prediction',dimensions = 10, gae_model_selection = 'gcn_ae')
    #pipeline.pipeline(dataset = '../../data/karate.txt', output = '../../embeddings/emb.txt', method = 'GAE', task = 'link-prediction',dimensions = 10, gae_model_selection = 'gcn_vae')


if __name__ == "__main__":
    more_main()
