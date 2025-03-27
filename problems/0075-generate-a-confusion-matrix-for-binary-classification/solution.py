
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
    TP = sum(row[0] == row[1] and row[0] == 1 for row in data)
    FN = sum(row[0] != row[1] and row[0] == 1 for row in data)
    FP = sum(row[0] != row[1] and row[0] == 0 for row in data)
    TN = sum(row[0] == row[1] and row[0] == 0 for row in data)
    return [[TP,FN],
            [FP,TN]]
    
