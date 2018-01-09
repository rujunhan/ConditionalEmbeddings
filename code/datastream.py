import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
from collections import OrderedDict
import time 
import random

class load_data():
    def __init__(self, args):

        self.batch_size = args.batch
        self.data_size = args.data_size
        self.label_map = args.label_map

        self.vocab = np.load(args.source+args.vocab).item()

        fname = args.source+args.source_file
        self.source = open(fname, 'r')

        self.end_of_data = False
        self.skips = args.skips
        self.labels = args.label_map.keys()

        self.count = 0

    def reset(self):
        self.source.seek(0)
        self.count = 0
        raise StopIteration

    def __iter__(self):
        return(self)

    def __next__(self):

        data = []
        count = 0

        while True:
            line = self.source.readline()

            if line == '':
                print("end of file!")
                self.reset()
                break

            self.count += 1
            line = line.strip().split('\t')

            label = line[0]
            if len(line) < 2:
                continue
            else:
                text = line[1]

            text_list = text.split(" ")

            if len(text_list) < self.skips * 2 + 1:
                continue

            elif label in self.labels:
                count += 1

            for i in range(self.skips, len(text_list) - self.skips):

                out_text = text_list[i-self.skips:i] + text_list[i+1:i+self.skips+1]

                in_text = text_list[i]

                data.append((label, in_text, out_text))

            if count >= self.batch_size:
                break

        in_idxs, out_idxs, covars = self.create_batch(data, self.vocab)

        return in_idxs, out_idxs, covars

    def create_batch(self, raw_batch, vocab):

        #input                                                                                  \
                                                                                                 
        all_txt = list(zip(*raw_batch))
        idxs = list(map(lambda w: vocab[w], all_txt[1]))
        in_idxs = Variable(torch.LongTensor(idxs).view(len(raw_batch),1), requires_grad=False)

        #output                                                                                 \
                                                                                                 
        idxs = list(map(lambda output: [vocab[w] for w in output], all_txt[2]))
        out_idxs = Variable(torch.LongTensor(idxs).view(len(raw_batch), -1), requires_grad=False)

        #covariates                                                                             \
                                                                                                 
        cvrs =  list(map(lambda c: self.label_map[c], all_txt[0]))
        cvrs = Variable(torch.LongTensor(cvrs).view(len(raw_batch), -1))
        return in_idxs, out_idxs, cvrs