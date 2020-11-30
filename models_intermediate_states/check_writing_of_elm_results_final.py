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
                                                                                                                                                                                     

#### ELM part

results_descriptors = [
	['raw', 'k00%d/epoched_kx_data_combined_out_rej_raw'],
	['Raw-CSP', 'k00%d/epoched_kx_data_combined_out_rej_CSP'],
	['Raw-CSP-Hil', 'k00%d/epoched_kx_data_combined_out_rej_CSP_hil'],
	['Raw-CCAr', 'k00%d/epoched_kx_data_combined_out_rej_CCAr'],
	['Raw-CCAR-Hil', 'k00%d/epoched_kx_data_combined_out_rej_CCAR_hil'],
	['Intrplt-raw', 'k00%d/epoched_intrplt_kx_data_combined_out_rej_raw'],
	['Intrplt-CSP', 'k00%d/epoched_intrplt_kx_data_combined_out_rej_CSP'],
	['Intrplt-CSP-Hil', 'k00%d/epoched_intrplt_kx_data_combined_out_rej_CSP_hil'],
	['Intrplt-CCAR', 'k00%d/epoched_intrplt_kx_data_combined_out_rej_CCAr'],
	['Intrplt-CCAR-Hil', 'k00%d/epoched_intrplt_kx_data_combined_out_rej_CCAR_hil'],
	['intrplt-le-100-raw', 'k00%d/epoched_intrplt_filt_under_100_kx_out_rej_raw'],
	['intrplt-le-100-CSP', 'k00%d/epoched_intrplt_filt_under_100_kx_out_rej_CSP'],
	['intrplt-le-100-CSP-Hil', 'k00%d/epoched_intrplt_filt_under_100_kx_out_rej_CSP_hil'],
	['intrplt-le-100-CCAr', 'k00%d/epoched_intrplt_filt_under_100_kx_out_rej_CCAr'],
	['intrplt-le-100-CCAr-Hil', 'k00%d/epoched_intrplt_filt_under_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-100-raw', 'k00%d/epoched_intrplt_filt_over_100_kx_out_rej_raw'],
	['intrplt-ge-100-CSP', 'k00%d/epoched_intrplt_filt_over_100_kx_out_rej_CSP'],
	['intrplt-ge-100-CSP-Hil', 'k00%d/epoched_intrplt_filt_over_100_kx_out_rej_CSP_hil'],
	['intrplt-ge-100-CCAr', 'k00%d/epoched_intrplt_filt_over_100_kx_out_rej_CCAr'],
	['intrplt-ge-100-CCAr-Hil', 'k00%d/epoched_intrplt_filt_over_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-400-raw', 'k00%d/epoched_intrplt_filt_over_400_kx_out_rej_raw'],
	['intrplt-ge-400-CSP', 'k00%d/epoched_intrplt_filt_over_400_kx_out_rej_CSP'],
	['intrplt-ge-400-CSP-Hil', 'k00%d/epoched_intrplt_filt_over_400_kx_out_rej_CSP_hil'],
	['intrplt-ge-400-CCAr', 'k00%d/epoched_intrplt_filt_over_400_kx_out_rej_CCAr'],
	['intrplt-ge-400-CCAr-Hil', 'k00%d/epoched_intrplt_filt_over_400_kx_out_rej_CCAR_hil'],
	['intrplt-500-900-raw', 'k00%d/epoched_intrplt_filt_500_900_kx_out_rej_raw'],
	['intrplt-500-900-CSP', 'k00%d/epoched_intrplt_filt_500_900_kx_out_rej_CSP'],
	['intrplt-500-900-CSP-Hil', 'k00%d/epoched_intrplt_filt_500_900_kx_out_rej_CSP_hil'],
	['intrplt-500-900-CCAr', 'k00%d/epoched_intrplt_filt_500_900_kx_out_rej_CCAr'],
	['intrplt-500-900-CCAr-Hil', 'k00%d/epoched_intrplt_filt_500_900_kx_out_rej_CCAR_hil']
]

results_aggregated = [
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[]
]

confusion_matrices = np.load('/media/christoph/Volume/Masterthesis/elm_models_trained/balanced_classes_confusion_matrices/confusion_matrices_run_01_on_data_preprocessed_final.npy', allow_pickle=True) 
for k in range(1, 11): 
#	file_for_subject = open('/media/christoph/Volume/Masterthesis/elm_models_trained/results/k00%d.txt' % k,'w+') 

	for elem in confusion_matrices: 
		if ('k00%d' % k) in elem[0] and not ('k00%d0' % k) in elem[0]:  
			title_loaded, confusion_matrix_loaded = elem     
			current_title = title_loaded
			print(current_title) 

			for idx, results_descriptors_tuple in enumerate(results_descriptors):

				if (results_descriptors_tuple[1] % k) in current_title and not 'z_normalized' in current_title and 'outliers_rejected' in current_title and not ((results_descriptors_tuple[1] % k) + '_hil') in current_title:

					pred_pos = confusion_matrix_loaded[0] 
					pred_neg = confusion_matrix_loaded[1] 
					tp, fn = pred_pos 
					fp, tn = pred_neg
					sensitivity, specificity, accuracy, f1_score, mcc = metrics_for_conf_mat(tn, fp, fn, tp)

					results_aggregated[idx].append(accuracy)

np.save('/media/christoph/Volume/Masterthesis/elm_models_trained/results/results_aggregated_outliers_rejected', results_aggregated)

#-#
#-#			pred_pos = confusion_matrix_loaded[0] 
#-#			pred_neg = confusion_matrix_loaded[1] 
#-#			tp, fn = pred_pos 
#-#			fp, tn = pred_neg
#-#			sensitivity, specificity, accuracy, f1_score, mcc = metrics_for_conf_mat(tn, fp, fn, tp)
#-#
#-#			file_for_subject.write(
#-#				'%s & %d & %d & %d & %d & %f & %f & %f & %f & %f \\\\\n\\hline\n' % (
#-#					current_title, tn, fp, fn, tp, sensitivity, specificity, accuracy, f1_score, mcc
#-#				)
#-#			) 
#-#	file_for_subject.close()



### alexnet part

from os import listdir

import numpy as np
from numpy import load

path = '/media/christoph/Volume/Masterthesis/alex_net/%d/'

results_descriptors = [
	['raw', 'k00%d-epoched_kx_data_combined_out_rej_raw'],
	['Raw-CSP', 'k00%d-epoched_kx_data_combined_out_rej_CSP'],
	['Raw-CSP-Hil', 'k00%d-epoched_kx_data_combined_out_rej_CSP_hil'],
	['Raw-CCAr', 'k00%d-epoched_kx_data_combined_out_rej_CCAr'],
	['Raw-CCAR-Hil', 'k00%d-epoched_kx_data_combined_out_rej_CCAR_hil'],
	['Intrplt-raw', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_raw'],
	['Intrplt-CSP', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CSP'],
	['Intrplt-CSP-Hil', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CSP_hil'],
	['Intrplt-CCAR', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CCAr'],
	['Intrplt-CCAR-Hil', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CCAR_hil'],
	['intrplt-le-100-raw', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_raw'],
	['intrplt-le-100-CSP', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CSP'],
	['intrplt-le-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CSP_hil'],
	['intrplt-le-100-CCAr', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CCAr'],
	['intrplt-le-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-100-raw', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_raw'],
	['intrplt-ge-100-CSP', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CSP'],
	['intrplt-ge-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CSP_hil'],
	['intrplt-ge-100-CCAr', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CCAr'],
	['intrplt-ge-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-400-raw', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_raw'],
	['intrplt-ge-400-CSP', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CSP'],
	['intrplt-ge-400-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CSP_hil'],
	['intrplt-ge-400-CCAr', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CCAr'],
	['intrplt-ge-400-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CCAR_hil'],
	['intrplt-500-900-raw', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_raw'],
	['intrplt-500-900-CSP', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CSP'],
	['intrplt-500-900-CSP-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CSP_hil'],
	['intrplt-500-900-CCAr', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CCAr'],
	['intrplt-500-900-CCAr-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CCAR_hil']
]

results_aggregated = [
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[]
]

for k in range(1, 11): 
#	file_for_subject = open('/media/christoph/Volume/Masterthesis/alex_net/results/k00%d.txt' % k,'w+') 

	for filename in listdir(path % k):
		if 'confusion_matrix' in filename:
			print(filename.replace('alex_net_on_', '').replace('_confusion_matrix.npy', ''))

			for idx, results_descriptors_tuple in enumerate(results_descriptors):
				current_title = filename.replace('alex_net_on_', '').replace('_confusion_matrix.npy', '')

				if (results_descriptors_tuple[1] % k) in current_title and not 'z_normalized' in current_title and 'outliers_rejected' in current_title and not ((results_descriptors_tuple[1] % k) + '_hil') in current_title:

					print('WOHOOO FIND ME')
					confusion_matrix_loaded = load((path % k) + filename, allow_pickle=True)

					pred_pos = confusion_matrix_loaded[0] 
					pred_neg = confusion_matrix_loaded[1] 
					tp, fn = pred_pos 
					fp, tn = pred_neg
					sensitivity, specificity, accuracy, f1_score, mcc = metrics_for_conf_mat(tn, fp, fn, tp)

					results_aggregated[idx].append(accuracy)

np.save('/media/christoph/Volume/Masterthesis/alex_net/results/results_aggregated_out_rej', results_aggregated)

#-#			file_for_subject.write(
#-#				'%s & %d & %d & %d & %d & %f & %f & %f & %f & %f \\\\\n\\hline\n' % (
#-#					filename.replace('alex_net_on_', '').replace('_confusion_matrix.npy', ''), tn, fp, fn, tp, sensitivity, specificity, accuracy, f1_score, mcc
#-#				)
#-#			) 
#-#
#-#	file_for_subject.close()


### mccnn part

from os import listdir

import numpy as np
from numpy import load

path = '/media/christoph/Volume/Masterthesis/mc_cnn/%d/'

results_descriptors = [
	['raw', 'k00%d-epoched_kx_data_combined_raw'],
	['Raw-CSP', 'k00%d-epoched_kx_data_combined_CSP'],
	['Raw-CSP-Hil', 'k00%d-epoched_kx_data_combined_CSP_hil'],
	['Raw-CCAr', 'k00%d-epoched_kx_data_combined_CCAr'],
	['Raw-CCAR-Hil', 'k00%d-epoched_kx_data_combined_CCAR_hil'],
	['Intrplt-raw', 'k00%d-epoched_intrplt_kx_data_combined_raw'],
	['Intrplt-CSP', 'k00%d-epoched_intrplt_kx_data_combined_CSP'],
	['Intrplt-CSP-Hil', 'k00%d-epoched_intrplt_kx_data_combined_CSP_hil'],
	['Intrplt-CCAR', 'k00%d-epoched_intrplt_kx_data_combined_CCAr'],
	['Intrplt-CCAR-Hil', 'k00%d-epoched_intrplt_kx_data_combined_CCAR_hil'],
	['intrplt-le-100-raw', 'k00%d-epoched_intrplt_filt_under_100_kx_raw'],
	['intrplt-le-100-CSP', 'k00%d-epoched_intrplt_filt_under_100_kx_CSP'],
	['intrplt-le-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_CSP_hil'],
	['intrplt-le-100-CCAr', 'k00%d-epoched_intrplt_filt_under_100_kx_CCAr'],
	['intrplt-le-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_CCAR_hil'],
	['intrplt-ge-100-raw', 'k00%d-epoched_intrplt_filt_over_100_kx_raw'],
	['intrplt-ge-100-CSP', 'k00%d-epoched_intrplt_filt_over_100_kx_CSP'],
	['intrplt-ge-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_CSP_hil'],
	['intrplt-ge-100-CCAr', 'k00%d-epoched_intrplt_filt_over_100_kx_CCAr'],
	['intrplt-ge-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_CCAR_hil'],
	['intrplt-ge-400-raw', 'k00%d-epoched_intrplt_filt_over_400_kx_raw'],
	['intrplt-ge-400-CSP', 'k00%d-epoched_intrplt_filt_over_400_kx_CSP'],
	['intrplt-ge-400-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_CSP_hil'],
	['intrplt-ge-400-CCAr', 'k00%d-epoched_intrplt_filt_over_400_kx_CCAr'],
	['intrplt-ge-400-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_CCAR_hil'],
	['intrplt-500-900-raw', 'k00%d-epoched_intrplt_filt_500_900_kx_raw'],
	['intrplt-500-900-CSP', 'k00%d-epoched_intrplt_filt_500_900_kx_CSP'],
	['intrplt-500-900-CSP-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_CSP_hil'],
	['intrplt-500-900-CCAr', 'k00%d-epoched_intrplt_filt_500_900_kx_CCAr'],
	['intrplt-500-900-CCAr-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_CCAR_hil']
]

results_aggregated = [
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[]
]

for k in range(1, 11): 
#	file_for_subject = open('/media/christoph/Volume/Masterthesis/alex_net/results/k00%d.txt' % k,'w+') 

	for filename in listdir(path % k):
		if 'confusion_matrix' in filename:
			print(filename.replace('mc_cnn_on_', '').replace('_confusion_matrix.npy', ''))

			for idx, results_descriptors_tuple in enumerate(results_descriptors):
				current_title = filename.replace('mc_cnn_on_', '').replace('_confusion_matrix.npy', '')

				if (results_descriptors_tuple[1] % k) in current_title and not 'z_normalized' in current_title and 'outliers_rejected' in current_title and not ((results_descriptors_tuple[1] % k) + '_hil') in current_title:

					print('WOHOOO FIND ME')
					confusion_matrix_loaded = load((path % k) + filename, allow_pickle=True)

					pred_pos = confusion_matrix_loaded[0] 
					pred_neg = confusion_matrix_loaded[1] 
					tp, fn = pred_pos 
					fp, tn = pred_neg
					sensitivity, specificity, accuracy, f1_score, mcc = metrics_for_conf_mat(tn, fp, fn, tp)

					results_aggregated[idx].append(accuracy)

np.save('/media/christoph/Volume/Masterthesis/mc_cnn/results/results_aggregated_out_rej', results_aggregated)





###########
###########

### deeprnn part

from os import listdir

import numpy as np
from numpy import load

path = '/media/christoph/Volume/Masterthesis/deep_rnn/%d/'

results_descriptors = [
	['raw', 'k00%d-epoched_kx_data_combined_out_rej_raw'],
	['Raw-CSP', 'k00%d-epoched_kx_data_combined_out_rej_CSP'],
	['Raw-CSP-Hil', 'k00%d-epoched_kx_data_combined_out_rej_CSP_hil'],
	['Raw-CCAr', 'k00%d-epoched_kx_data_combined_out_rej_CCAr'],
	['Raw-CCAR-Hil', 'k00%d-epoched_kx_data_combined_out_rej_CCAR_hil'],
	['Intrplt-raw', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_raw'],
	['Intrplt-CSP', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CSP'],
	['Intrplt-CSP-Hil', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CSP_hil'],
	['Intrplt-CCAR', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CCAr'],
	['Intrplt-CCAR-Hil', 'k00%d-epoched_intrplt_kx_data_combined_out_rej_CCAR_hil'],
	['intrplt-le-100-raw', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_raw'],
	['intrplt-le-100-CSP', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CSP'],
	['intrplt-le-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CSP_hil'],
	['intrplt-le-100-CCAr', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CCAr'],
	['intrplt-le-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_under_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-100-raw', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_raw'],
	['intrplt-ge-100-CSP', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CSP'],
	['intrplt-ge-100-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CSP_hil'],
	['intrplt-ge-100-CCAr', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CCAr'],
	['intrplt-ge-100-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_100_kx_out_rej_CCAR_hil'],
	['intrplt-ge-400-raw', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_raw'],
	['intrplt-ge-400-CSP', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CSP'],
	['intrplt-ge-400-CSP-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CSP_hil'],
	['intrplt-ge-400-CCAr', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CCAr'],
	['intrplt-ge-400-CCAr-Hil', 'k00%d-epoched_intrplt_filt_over_400_kx_out_rej_CCAR_hil'],
	['intrplt-500-900-raw', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_raw'],
	['intrplt-500-900-CSP', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CSP'],
	['intrplt-500-900-CSP-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CSP_hil'],
	['intrplt-500-900-CCAr', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CCAr'],
	['intrplt-500-900-CCAr-Hil', 'k00%d-epoched_intrplt_filt_500_900_kx_out_rej_CCAR_hil']
]

results_aggregated = [
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[],
	[]
]

for k in range(1, 11): 
#	file_for_subject = open('/media/christoph/Volume/Masterthesis/alex_net/results/k00%d.txt' % k,'w+') 

	for filename in listdir(path % k):
		if 'confusion_matrix' in filename:
			print(filename.replace('deep_rnn_on_', '').replace('_confusion_matrix.npy', ''))

			for idx, results_descriptors_tuple in enumerate(results_descriptors):
				current_title = filename.replace('deep_rnn_on_', '').replace('_confusion_matrix.npy', '')

				if (results_descriptors_tuple[1] % k) in current_title and not 'z_normalized' in current_title and 'outliers_rejected' in current_title and not ((results_descriptors_tuple[1] % k) + '_hil') in current_title:

					print('WOHOOO FIND ME')
					confusion_matrix_loaded = load((path % k) + filename, allow_pickle=True)

					pred_pos = confusion_matrix_loaded[0] 
					pred_neg = confusion_matrix_loaded[1] 
					tp, fn = pred_pos 
					fp, tn = pred_neg
					sensitivity, specificity, accuracy, f1_score, mcc = metrics_for_conf_mat(tn, fp, fn, tp)

					results_aggregated[idx].append(accuracy)

np.save('/media/christoph/Volume/Masterthesis/deep_rnn/results/results_aggregated_out_rej', results_aggregated)