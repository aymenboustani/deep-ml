import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
	"""
	Compute the GRPO objective function.

	Args:
		rhos: List of likelihood ratios (p_i) = pi_theta(o_i | q) / pi_theta_old(o_i | q).
		A: List of advantage estimates (A_i).
		pi_theta_old: List representing the old policy probabilities pi_theta_old(o_i | q).
		pi_theta_ref: List representing the reference policy probabilities pi_ref(o_i | q).
		epsilon: Clipping parameter (eps).
		beta: KL divergence penalty coefficient (beta).

	Returns:
		The computed GRPO objective value.
	"""
	# Your code here
	G = len(A)
	pi_theta_ref = np.array(pi_theta_ref)
	pi_theta = np.array(rhos) * np.array(pi_theta_old)
	pi_theta /= np.sum(pi_theta)
	pi_theta_ref /= np.sum(pi_theta_ref)
	DKL = sum(p * np.log(p / ref + 1e-10) for p, ref in zip(pi_theta, pi_theta_ref))
	surr = sum(min(rho * a, a * np.clip(rho, 1 - epsilon, 1 + epsilon)) for rho, a in zip(rhos, A)) / G
	return surr - beta * DKL