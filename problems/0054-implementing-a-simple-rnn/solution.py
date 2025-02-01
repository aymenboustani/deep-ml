import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	h_t = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)
	for i in range(len(input_sequence)):
		x_t = input_sequence[i]
		h_t = np.tanh(np.dot(Wx, x_t)+ np.dot(Wh, h_t) + b)
	return h_t.tolist()