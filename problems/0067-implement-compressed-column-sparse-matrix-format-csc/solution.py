def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense = [(dense_matrix[i][j], i , j) for j in range(len(dense_matrix[0])) for i in range(len(dense_matrix)) if dense_matrix[i][j] != 0]
	
	return [value[0] for value in dense], [value[1] for value in dense], [value[2] for value in dense]


