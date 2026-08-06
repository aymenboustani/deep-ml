import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	grad = np.array(gradient)
	magnitude = np.linalg.norm(grad)
	unit = grad / magnitude if magnitude != 0 else np.zeros(*grad.shape)
	return {
		'magnitude':magnitude,
		'direction':unit,
		'descent_direction':-unit
	}