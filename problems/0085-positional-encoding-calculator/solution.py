import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
    if position == 0 or d_model <= 0 : return -1
    pos_encoding = np.zeros((position, d_model))
    for i in range(position):
        for j in range(d_model):
            if j % 2 == 0:
                pos_encoding[i,j] = np.sin(i / (10000 ** (j / d_model)))
            else:
                pos_encoding[i,j] = np.cos(i / (10000 ** ((j - 1) / d_model)))
	pos_encoding = np.float16(pos_encoding)
	return pos_encoding