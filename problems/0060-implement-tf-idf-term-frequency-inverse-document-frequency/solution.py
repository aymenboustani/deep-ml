import numpy as np

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
    results = []
    N = len(corpus)
    DFs = {q : sum(q in doc for doc in corpus) for q in query}
    IDFs = {q : np.log((N + 1) / (df +1)) + 1 for q, df in DFs.items()}
    for doc in corpus:
        scores = []
        for q in query:
            TF = doc.count(q) / len(doc)
            scores.append(TF * IDFs[q])
        results.append(scores)
    return results

