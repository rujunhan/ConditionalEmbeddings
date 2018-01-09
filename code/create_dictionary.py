import numpy as np
import os
import time
from collections import OrderedDict
#import spacy
import argparse
import json
import re
#nlp = spacy.load('en')


def process(text):
    '''
    processed = []
    for i in nlp(text):
        if i.pos_ in ['SPACE', 'PUNCT']:
            continue
        elif i.pos_ == 'PART':
            processed.append('_s')
        elif i.pos_ in ['NUM', 'SYM']:
            processed.append(i.pos_)
        else:
            processed.append(i.text)
    '''

    processed = re.sub(r'[^a-zA-Z ]',r' ', text.replace('-\n','').replace('\n',' ')).lower().split()
    return processed

def main(args):

    vocab0 = OrderedDict()
    start = time.time()

    all_files = [x for x in os.listdir(args.source) if '.json' in x]
    print(all_files)

    with open('%s%s.txt'%(args.saveto, args.save_label), 'w') as f:
        print("loding all files...")

        for file_name in all_files:
            print(file_name)
            
            file = open(args.source+file_name)
            for line in file:
                d = json.loads(line)
                text = d['text']
                if text != None:
                    label = d['filedate'][:3]

                    words = process(text)
                    if len(words) < args.window:
                        continue

                    f.write(label+"\t")
                    for w in words:
                        f.write(w+" ")
                        if w in vocab0:
                            vocab0[w] += 1
                        else:
                            vocab0[w] = 1
                    f.write("\n")
            print("%s seconds elapsed" % (time.time() - start))
    f.close()
    
    tokens = list(vocab0.keys())
    
    freqs = list(vocab0.values())

    sidx = np.argsort(freqs)[::-1]
    vocab = OrderedDict([(tokens[s],i) for i, s in enumerate(sidx)])
    
    #vocab_f = OrderedDict({k: (vocab0[k]/total_words)**(3/4) for k in vocab.keys()})

    np.save(args.saveto+"vocab_f"+args.save_label+".npy",vocab0)
    np.save(args.saveto+"vocab"+args.save_label+".npy", vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', type=str, default="")
    parser.add_argument('-saveto', type=str, default="")
    parser.add_argument('-text', type=str, default="")
    parser.add_argument('-window',type=int, default=7)
    parser.add_argument('-save_label', type=str, default="")

    args = parser.parse_args()
    main(args)




