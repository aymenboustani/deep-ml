import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	PMI = np.log2(joint_counts * total_samples / total_counts_x / total_counts_y)
	if PMI.is_integer():
        return int(PMI)
    else:
        return np.round(PMI, 3)