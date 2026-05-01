import numpy as np

def mtp_loss(main_logits, main_targets, mtp_logits, mtp_targets, mtp_weight):
    """
    Compute the combined LM + depth-1 Multi-Token Prediction loss.

    Args:
        main_logits: array-like (N, V)
        main_targets: array-like (N,) integer token ids
        mtp_logits: array-like (N-1, V)
        mtp_targets: array-like (N-1,) integer token ids
        mtp_weight: float

    Returns:
        float: total loss rounded to 6 decimals
    """
    def softmax(x):
        x = x - np.max(x, axis = 1, keepdims = True)
        x = np.exp(x)
        return x / x.sum(axis = 1, keepdims = True)

    
    N = len(main_logits)
    main_logits = softmax(main_logits)
    mtp_logits = softmax(mtp_logits)
    main_logprob = np.log(main_logits[np.arange(N), main_targets])
    mtp_logprob = np.log(mtp_logits[np.arange(N - 1), mtp_targets])
    LM = - 1 / N * np.sum(main_logprob)
    LMTP =  1 / (1 - N) * np.sum(mtp_logprob)
    L = LM + mtp_weight * LMTP
    return np.round(L, 6)
