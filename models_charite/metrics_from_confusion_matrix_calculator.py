import numpy as np

def metrics_for_conf_mat(tn, fp, fn, tp):
	"""
	Computes for given confusion-matrix entries the metrics
	Sensitivity, Specificity, Accuracy, F1-Score and MCC 
	More info: https://en.wikipedia.org/wiki/Confusion_matrix
	"""
	sensitivity = tp / (tp + fn)
	specificity = tn / (tn + fp)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	f1_score = (2 * tp) / ((2 * tp) + fp + fn)
	mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	return (sensitivity, specificity, accuracy, f1_score, mcc)