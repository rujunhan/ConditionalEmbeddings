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

class ConditinalBBP(nn.Module):
    def __init__(self, num_words, embed_size, args, weights=None):
        super(ConditinalBBP, self).__init__()

        self.num_words = num_words
        self.embed_size = embed_size
        self.f = args.function
        self.label_map = args.label_map

        self.n_labels = len(list(self.label_map.keys()))
        self.num_sampled = args.negs
        self.pr_w = args.prior_weight
        self.s1 = args.sigma_1
        self.s2 = args.sigma_2
        
        ### mu  
        self.out_embed = nn.Embedding(num_words, self.embed_size,sparse=True)
        
        self.out_embed.weight = Parameter(torch.FloatTensor(num_words, self.embed_size).uniform_(-1, 1))

        self.in_embed = nn.Embedding(num_words, self.embed_size,sparse=True)
        
        self.in_embed.weight = Parameter(torch.FloatTensor(num_words, self.embed_size).uniform_(-1, 1))
        
        ### rho
        self.out_rho = nn.Embedding(num_words, self.embed_size,sparse=True)
        
        self.out_rho.weight = Parameter(torch.FloatTensor(num_words, self.embed_size).uniform_(-1, 1))

        self.in_rho = nn.Embedding(num_words, self.embed_size,sparse=True)
        
        self.in_rho.weight = Parameter(torch.FloatTensor(num_words, self.embed_size).uniform_(-1, 1))
        
        ### covariance
        self.covariates = nn.Embedding(self.n_labels, self.embed_size)
        
        self.covariates.weight = Parameter(torch.FloatTensor(self.n_labels, self.embed_size).uniform_(-1, 1))

        self.linear = nn.Linear(embed_size*2, embed_size)
        self.act = nn.Tanh()

        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"

            self.weights = Variable(torch.from_numpy(weights)).float()

    def sample_var_noise(self,v):
        n = v.size()[0]
        std_tsr = torch.ones(v.size())
        return Variable(torch.normal(mean=0, std=std_tsr)).float()

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights 
        """
        return torch.multinomial(self.weights, num_sample, replacement=True)

    def reshape(self, v, win):
        return v.expand(v.size()[0], win, v.size()[2]).contiguous().view(-1, self.embed_size)

    def compute_prior(self, w):
        n1 = self.pr_w*(-w**2/(2*self.s1**2)).exp()#/(math.sqrt(2*math.pi)*self.s1)
        n2 = (1-self.pr_w)*(-w**2/(2*self.s2**2)).exp()              #/(math.sqrt(2*math.pi)*self.s2)
        return (n1+n2).log().sum(1)

    def forward(self, inputs, outputs, covars, wt):


        use_cuda = self.out_embed.weight.is_cuda
        
        [batch_size, window_size] = outputs.size()
        
        # y is the covariate vector, should have the same size as word vector
        y = self.covariates(covars.repeat(1, window_size).contiguous().view(-1))
        
        ### mu_in: (window_size * batch) * embed_size
        mu_in = self.in_embed(inputs)
        eps_in = self.sample_var_noise(mu_in)
        
        mu_in = self.reshape(mu_in, window_size)
        eps_in = self.reshape(eps_in, window_size)
        
        ### sigma_in
        sig_in = (self.in_rho(inputs).exp() + 1).log()
        sig_in = self.reshape(sig_in, window_size)
        
        ### weights_in
        if use_cuda:
            eps_in = eps_in.cuda()
        
        w_in = self.act(self.linear(torch.cat([mu_in, y], 1))) + sig_in * eps_in

        post_in = -0.5 * (eps_in**2).sum(1) - sig_in.log().sum(1) #- math.log(math.sqrt((2*math.pi)**self.embed_size))

        prior_in = self.compute_prior(w_in)

        ### mu_out: (window_size * batch) * embed_size
        mu_out = self.out_embed(outputs)
        eps_out = self.sample_var_noise(mu_out)
        mu_out = self.reshape(mu_out, window_size)

        eps_out = self.reshape(eps_out, window_size)

        ### sigma_out
        sig_out = (self.out_rho(outputs).exp() + 1).log()
        sig_out = self.reshape(sig_out, window_size)

        if use_cuda:
            eps_out = eps_out.cuda()

        w_out = mu_out + sig_out * eps_out
        
        mu_out = self.out_embed(outputs.contiguous().view(-1))

        post_out = -0.5*(eps_out **2).sum(1) - sig_out.log().sum(1) #- math.log(math.sqrt((2*math.pi)**self.embed_size))
        prior_out = self.compute_prior(w_out)
        
        log_target = (w_in * w_out).sum(1).sigmoid().log()
        
        if self.weights is not None:
            noise_sample_count = batch_size * self.num_sampled
            draw = self.sample(noise_sample_count)
            
            noise = draw.view(batch_size, self.num_sampled)
            
        else:
            noise = Variable(torch.Tensor(batch_size, self.num_sampled).
                             uniform_(0, self.num_words-1).long())
        if use_cuda:
            noise = noise.cuda()
        
        noise = self.out_embed(noise).neg().view(-1,self.embed_size)
        
        log_sampled = (w_in * noise).sum(1).sigmoid().log()

        likelihood = log_target + log_sampled

        loss = wt*(post_in + post_out - prior_in - prior_out) - likelihood
        return loss.mean()
        
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

    def covar_embeddings(self):
        return self.covariates.weight.data.cpu().numpy()

    def var_embeddings(self):
        return self.in_rho.weight.data.cpu().numpy()
