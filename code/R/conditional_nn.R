dir = ""
setwd(dir)

library(RcppCNPy)
library("rjson")
library("lsa")

## load conditional embedding model parameters and dictionary
emb = npyLoad("us_speech/results/emb.npy")
cvr = npyLoad("us_speech/results/cvr.npy")
var = npyLoad("us_speech/results/var.npy")
W = npyLoad("us_speech/results/W.npy")
b = npyLoad("us_speech/results/b.npy")
count_table = fromJSON(file='us_speech/results/count_table.json')
vocab = fromJSON(file = 'us_speech/results/vocab.json')
for (x in names(vocab)) {vocab[x] = (vocab[x][[1]] + 1)}


# reproduce the MLP layer in the model
MLP_trans <- function(mu, cvr, w, b){
  temp = c(mu, cvr)
  return(tanh(w %*% as.matrix(temp) + b))
}


KL <- function(mu1, cov1, w2, emb, var, cvr, w, b, vocab, cvr_idx){
  
  idx = vocab[w2][[1]]
  # transform mean based on covariate
  mu2 = emb[idx, ]
  mu2 = MLP_trans(mu2, cvr[cvr_idx, ], w, b)
  
  # construct covariance matrix
  cov2 = log(1 + exp(var[idx, ]))
  
  return(sum((mu1 - mu2)^2 / (2 * cov2^2) + 1.0/2.0 * ((cov1/cov2)^2 - 1.0 - log((cov1/cov2)^2))))
}

cosine_sim <- function(mu1, w2, emb, cvr, w, b, vocab, cvr_idx){
  
  idx = vocab[w2][[1]]
  # transform mean based on covariate
  mu2 = emb[idx, ]
  mu2 = MLP_trans(mu2, cvr[cvr_idx, ], w, b)
  #print(mu1)
  #print(mu2)
  return(cosine(as.vector(mu1), as.vector(mu2)))
}

nearest_neighbors <- function(query, emb, var, cvr, w, b, top_n, vocab, cvr_idx, func){
  
  idx = vocab[query][[1]]
  
  # transform mean based on covariate
  mu1 = emb[idx, ]
  mu1 = MLP_trans(mu1, cvr[cvr_idx, ], w, b)
  
  # construct covariance matrix
  cov1 = log(1 + exp(var[idx, ]))
  
  keys = names(vocab)
  
  # compute word distance based on different metrics
  sims = numeric(length(keys))
  if (func == 'KL') {
    for (k in 1:length(keys)){
      sims[k] = KL(mu1, cov1, keys[k], emb, var, cvr, w, b, vocab, cvr_idx)
      rev = FALSE
    }
  }
  else{
    if (func == 'cosine'){
      for (k in 1:length(keys)){
        sims[k] = cosine_sim(mu1, keys[k], emb, cvr, w, b, vocab, cvr_idx)
        rev = TRUE
      }
    }
  }
  
  # retrieve nearest words based on sorted results
  nn = character(top_n)
  res = sort(sims, decreasing = rev, index.return = TRUE)
  top_idx = res$ix[1:top_n]
  
  for (n in 1:top_n){
    nn[n] = keys[top_idx[n]]
  }
  
  return(nn)
}

Hotelling_T2 <- function(m1, m2, s1, s2, n1, n2){

  if (n1 == 0 | n2 == 0){
    T2 = 0
  }
  else{
    if (n1 + n2 <= 2){
      T2 = 0
    }
    else{
      s = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
      T2 = sum((m1 - m2)^2 / s)
    }
  }
  return(T2)
}

T_test <- function(w1, w2, emb, var, cvr, w, b, vocab, cvr_idx, count_table){
  
  # mean
  idx1 = vocab[w1][[1]]
  m1 = emb[idx1, ]
  m1 = MLP_trans(m1, cvr[cvr_idx, ], w, b)
  
  idx2 = vocab[w2][[1]]
  m2 = emb[idx2, ]
  m2 = MLP_trans(m2, cvr[cvr_idx, ], w, b)
  
  # varianc
  s1 = (log(1 + exp(var[idx1, ])))^2
  s2 = (log(1 + exp(var[idx2, ])))^2
  
  # counts
  n1 = count_table[w1][[1]][cvr_idx]
  n2 = count_table[w2][[1]][cvr_idx]
  
  return(Hotelling_T2(m1, m2, s1, s2, n1, n2))
}

query = 'healthcare'
top_n = 50
label_map = list('R', 'D', 'O')
cvr_idx = which(label_map == 'R')


# obtain nearest neighbors
nn = nearest_neighbors(query, emb, var, cvr, W, b, top_n, vocab, cvr_idx, 'cosine')

# T-test
for (n in nn){
  print(paste(n, toString(T_test(query, n, emb, var, cvr, W, b, vocab, cvr_idx, count_table)), sep=":     "))
}



