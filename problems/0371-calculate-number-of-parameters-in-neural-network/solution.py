def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
	params = 0
	for layer in layers:
		if 'bias' not in layer.keys():
			layer['bias'] = True
		if layer['type'] == 'dense':
			params += (layer['input_size'] + 1) * layer['output_size'] if layer['bias'] else (layer['input_size']) * layer['output_size']
		else:
			params += (layer['kernel_size'] ** 2 * layer['in_channels'] + 1) * layer['out_channels'] if layer['bias'] else (layer['kernel_size'] ** 2 * layer['in_channels']) * layer['out_channels']
	return params

