dir = ""
setwd(dir)

library(RcppCNPy)
library("rjson")
library("lsa")
source('utils.R')

## load conditional embedding model parameters and dictionary
emb = npyLoad("us_speech/results/emb.npy")
cvr = npyLoad("us_speech/results/cvr.npy")
var = npyLoad("us_speech/results/var.npy")
W = npyLoad("us_speech/results/W.npy")
b = npyLoad("us_speech/results/b.npy")
count_table = fromJSON(file='us_speech/results/count_table.json')
vocab = fromJSON(file = 'us_speech/results/vocab.json')
for (x in names(vocab)) {vocab[x] = (vocab[x][[1]] + 1)}


### Specify hyper-parameters
query = 'healthcare'
top_n = 50
label_map = list('R', 'D', 'O')
cvr_idx = which(label_map == 'R')

### obtain nearest neighbors
nn = nearest_neighbors(query, emb, var, cvr, W, b, top_n, vocab, cvr_idx, 'cosine')

### T-test
for (n in nn){
  print(paste(n, toString(T_test(query, n, emb, var, cvr, W, b, vocab, cvr_idx, count_table)), sep=":     "))
}



