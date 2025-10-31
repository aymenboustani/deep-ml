# Implement your function below.

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Your code here
    words_ref = reference.split()
    words_can = candidate.split()
    overlap = sum(a == b for a, b in zip(words_ref, words_can))
    P = overlap / len(words_can)
    R = overlap / len(words_ref)
    F1 = 2 * P * R / (P + R)
    return {"precision": P,
            "recall" : R,
            "f1" : F1}
