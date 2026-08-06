import numpy as np

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value
	"""
	# Your code here
	def softmax(x):
		return np.exp(x) / np.exp(x).sum(keepdims=True)
	
	T=temperature
	
	return T ** 2 * np.sum(softmax(teacher_logits / T) * np.log(softmax(teacher_logits / T)/softmax(student_logits / T)))