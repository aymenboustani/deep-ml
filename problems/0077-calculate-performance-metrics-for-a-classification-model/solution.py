
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
    TP = sum(a == p == 1 for a, p in zip(actual, predicted))
    FP = sum(a != p and a == 0 for a, p in zip(actual, predicted))
    TN = sum(a == p == 0 for a, p in zip(actual, predicted))
    FN = sum(a != p and a == 1 for a, p in zip(actual, predicted))
    confusion_matrix = [[TP, FN],
                        [FP, TN]]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    negativePredictive = TN / (TN + FN)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * recall / (precision + recall)
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
