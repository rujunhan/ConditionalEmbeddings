import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import os
import math
from collections import OrderedDict
import time 
import matplotlib.pyplot as plt
import argparse
import random
import shutil
from BBP import ConditinalBBP
from datastream import load_data
from utils import *


def main(args):

    embedding_size = args.emb
    batch_size = args.batch  
    
    vocab = np.load(args.source+args.vocab).item()
    
    data_size = args.data_size

    n_words = len(vocab)

    model = ConditinalBBP(n_words, embedding_size, args)

    batch = load_data(args)

    n_batch = int(np.ceil(data_size/batch_size))
    print(n_batch)
 
    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    losses = []
    
    print("start training model...\n")
    start_time = time.time()
    
    if args.load_model:
        model, optimizer, loss, epoch = load_checkpoint(model, optimizer, args.saveto + args.best_model)
        print("train %s epochs before, loss is %s" % (epoch, loss))


    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0

        for in_v,out_v,cvrs in batch:
            w = batch_size / data_size

            if args.cuda:
                in_v, out_v, cvrs = in_v.cuda(), out_v.cuda(), cvrs.cuda()

            if batch.count % 100000 == 0:
                print("training epoch %s: completed %s %%"  % (str(epoch), str(round(100*batch.count/data_size, 2))))

            model.zero_grad()
            loss = model(in_v,out_v,cvrs, w)
            loss.backward()
            optimizer.step()
            total_loss+=loss.data.cpu().numpy()[0]

        ave_loss = total_loss/n_batch
        print("average loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        end_time = time.time()
        print("%s seconds elapsed" % str(end_time - start_time))

        is_best = False

        if epoch == 0:
            is_best = True
        elif ave_loss < losses[epoch-1]:
            is_best = True
            
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'loss': ave_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.saveto+args.best_model)
        
    print(losses)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', type=int, default=50)
    parser.add_argument('-batch', type=int, default=1)
    parser.add_argument('-n_epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default =0.05)
    parser.add_argument('-skips', type=int, default = 3)
    parser.add_argument('-negs', type=int, default = 6)
    parser.add_argument('-vocab', type=str)
    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)
    parser.add_argument('-file_list', type=str)
    parser.add_argument('-file_stamp', type=str)
    parser.add_argument('-save_stamp', type=str)
    parser.add_argument('-source_file', type=str)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-best_model_save_file', type=str, default='model_best.pth.tar')
    parser.add_argument('-label_map', type=list)
    parser.add_argument('-window',type=int, default=7)
    parser.add_argument('-prior_weight', type=float, default=0.5)
    parser.add_argument('-sigma_1', type=float, default=1)
    parser.add_argument('-sigma_2', type=float, default=0.2)
    parser.add_argument('-weight_scheme', type=int, default=1)
    parser.add_argument('-load_model', type=float, default=False)
    args = parser.parse_args()
    
    args.saveto = "../results/"
    args.source = "../uk_speech/data/"

    args.file_stamp = '1227_freq'
    args.vocab = "vocab%s.npy"%args.file_stamp
    args.save_stamp = '1227_freq'
    args.function = "NN"
    args.source_file = 'uk_%s.txt'%args.file_stamp

    args.data_size = 4075701

    args.file_list = "all_files"+args.file_stamp+".npy"
    args.cuda = False

    args.best_model_save_file = "model_best_%s.pth.tar"%args.save_stamp

    args.label_map = {str(v):k for k,v in enumerate(range(193, 202))}
    
    main(args)
    
   
