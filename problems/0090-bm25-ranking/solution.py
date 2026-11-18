import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    # Your code here
    count = lambda t, d : sum(term == t for term in d)
    N = len(corpus)
    avgdl = sum(len(doc) for doc in corpus) / N
    nt = [sum(term in doc for doc in corpus) for term in query]
    scores = []
    for doc in corpus:
        score = sum(
            np.log((N+1)/(nt[i]+1)) * count(t, doc) * (k1 + 1) / (count(t,doc) + k1 * (1 + b * (len(doc) / avgdl - 1))) for i, t in enumerate(query) 
            )
        scores.append(score)
    return np.round(scores,3)