def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    Xx = sum(x == xn for (xn, yn) in data)
    return round(sum(x == xn and y == yn for (xn, yn) in data) / Xx, 4) if Xx > 0 else 0.0