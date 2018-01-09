import numpy as np
import os
import time
from collections import OrderedDict
#import spacy
import argparse
import json
import re
#nlp = spacy.load('en')

def main(args):
    
    vocab = np.load(args.vocab).item()
    vocab_freq = OrderedDict([(i, s) for i, s in vocab.items() if s < args.top_K])
    freq_keys = vocab_freq.keys()
    
    vocab_t = np.load(args.vocab_f).item()
    vocab_t_freq = OrderedDict([(k, v) for k, v in vocab_t.items() if k in freq_keys])
    total_words = np.sum(list(vocab_t_freq.values()))
    print(total_words)

    vocab_p = OrderedDict([(k, 1 - np.sqrt(10**(-5) / (v / total_words))) for k, v in vocab_t_freq.items()])

    start = time.time()

    lines = open(args.source+'%s.txt'%args.save_label, 'r')
    
    with open(args.saveto+'%s_freq.txt'%args.save_label, 'w') as f:
        print("loding all files...")

        for line in lines:
            items = line.strip().split("\t")
            label = items[0][:3]
            text = items[1]
                
            if text != None:
                words = text.split(' ')
                
                f.write(label+"\t")
                
                for w in words:
                    if (w in freq_keys) and (np.random.uniform(0, 1) < vocab_p[w]):
                        f.write(w+" ")

                f.write("\n")

    print("%s seconds elapsed" % (time.time() - start))
    f.close()

    np.save(args.saveto+"vocab"+args.save_label+"_freq.npy", vocab_freq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', type=str, default="")
    parser.add_argument('-saveto', type=str, default="")
    parser.add_argument('-text', type=str, default="")
    parser.add_argument('-vocab', type=str, default="")
    parser.add_argument('-vocab_f', type=str, default="")
    parser.add_argument('-window',type=int, default=7)
    parser.add_argument('-save_label', type=str, default="") # file name of the saved files
    parser.add_argument('-top_K', type=int, default=25000)

    args = parser.parse_args()
    main(args)




