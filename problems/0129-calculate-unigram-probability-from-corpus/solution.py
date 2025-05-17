def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    words = corpus.split()
    return words.count(word) / len(words)