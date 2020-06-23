from config import Config
from graph import Graph
import scipy.io as sio
import time
from optparse import OptionParser
import os
from model.sdne import SDNE
import numpy as np
import sys

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = OptionParser()
    parser.add_option("-c", dest = "config_file", action = "store", metavar = "CONFIG FILE")
    options, _ = parser.parse_args()

    if options.config_file is None:
        raise IOError("no config file specified")

    config = Config(options.config_file)
    train_graph_data = Graph(config.train_graph_file, config.ng_sample_ratio, config.T, config.tao,
                             config.walk_times, config.walk_length, config.restart_ratio)

    if config.origin_graph_file:
        origin_graph_file = Graph(config.origin_graph_file, config.ng_sample_ratio)

    config.struct[0] = train_graph_data.N

    model = SDNE(config)
    model.do_variables_init(train_graph_data)

    epochs = 0
    batch_n = 0

    walks = train_graph_data.walks

    while(True):
        '''
        one epoch train
        '''
        for walk in walks:
            mini_batch = train_graph_data.get_mini_batch_by_walk(walk)
            model.fit(mini_batch)
        '''
        if one epoch train is over, then print the train loss
        '''
        loss = 0
        while(True):
            mini_batch = train_graph_data.sample(config.batch_size, do_shuffle=False)
            loss += model.get_loss(mini_batch)
            if train_graph_data.is_epoch_end:
                break
        print("Epoch : %d loss : %.3f" % (epochs, loss))
        if epochs == config.epochs_limit:
            print("exceed epochs limit terminating")
            break
        epochs += 1

    """
    model training is over, then get the embedding result
    """
    embedding = None
    while(True):
        mini_batch = train_graph_data.sample(config.batch_size, do_shuffle= False)
        if embedding is None:
            embedding = model.get_embedding(mini_batch)
        else:
            embedding = np.vstack((embedding, model.get_embedding(mini_batch)))
        if train_graph_data.is_epoch_end:
            break

    with open(config.embedding_filename, 'w') as fout:
        print(config.embedding_filename)
        for i in range(len(embedding)):
            fout.write(str(i) + ' ')
            for j in range(len(embedding[i])):
                fout.write(str(embedding[i][j]) + ' ')
            fout.write('\n')
        fout.close()
