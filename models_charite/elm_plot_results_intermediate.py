import sys

import matplotlib.pyplot as plt
import numpy as np

from numpy import load

# Path should be: '/home/christoph/Desktop/Beginning_September_Work/scripts/confusion_matrices_run_1_on_data_preprocessed_final.npy'
results_path = str(sys.argv[1])
test_confusion_matrix = load(results_path, allow_pickle=True)

from matplotlib.font_manager import FontProperties
from matplotlib import colors
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('oblique')
font.set_size(15)
ticks = np.arange(0, 1.1, 0.1)
ticklabels = [str(int(tick)) for tick in ticks * 100]

snnrs_path = '/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/snnrs.npy'
snnrs_of_subjects = list()

for i in range(1, 11):
    snnrs_of_subjects.append(load(snnrs_path % i)[0, 0])

snnrs_of_subjects = np.asarray(snnrs_of_subjects)

for j in range(0, 55, 1):
    hist_dat_per_sbuject = [] 
    current_title = ''
    print(j)
    for i in range(0 + j, len(test_confusion_matrix), 55): 
        subject_data = test_confusion_matrix[i] 
        title_loaded, confusion_matrix_loaded = subject_data[0], subject_data[1]     
        current_title = title_loaded
        print(current_title) 
         
        pred_pos = confusion_matrix_loaded[0] 
        pred_neg = confusion_matrix_loaded[1] 
        tp, fn = pred_pos 
        fp, tn = pred_neg 
        detection_rate = tp/(tp+fn) 
        positive_predictive_value = tp/(tp+fp) 
        matt_corr_eff = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))    
         
        hist_dat_per_sbuject.append(np.asarray([tp, fn, fp, tn, detection_rate, positive_predictive_value, matt_corr_eff])) 
        #print(confusion_matrix_loaded) 
        #print('Calculated DR_[%f], PPV_[%f], MCC_[%f] from confusion matrix above' % (detection_rate, positive_predictive_value, matt_corr_eff)) 
         
    hist_dat_per_sbuject = np.asarray(hist_dat_per_sbuject).flatten() 
     
    fig, ax = plt.subplots(figsize=(18,9))
    ax.scatter(snnrs_of_subjects, hist_dat_per_sbuject[np.arange(4, 71, 7)], label='DR (%)', color='blue', linewidth=5) 
    ax.scatter(snnrs_of_subjects, hist_dat_per_sbuject[np.arange(5, 71, 7)], label='PPV (%)', color='orange', linewidth=5) 
    ax.scatter(snnrs_of_subjects, hist_dat_per_sbuject[np.arange(6, 71, 7)], label='MCC (abs)', color='black', linewidth=5)
    
    twin_y = ax.twinx()

    fc = colors.to_rgba('lightgrey')
    ec = colors.to_rgba('black')
    fc = fc[:-1] + (0.7,) # <--- Change the alpha value of facecolor to be 0.4

    annotation_offsets_y = [15, 15, 15, 15, 35, 15, 25, 15, 35, 25]
    for i in range(1, 11):
        ax.annotate('K00%d' % i, (snnrs_of_subjects[i - 1], hist_dat_per_sbuject[6 + ((i - 1) * 7)]), xytext=(0, annotation_offsets_y[i - 1]), ha='center', textcoords='offset points', fontsize=15, bbox=dict(pad=2, facecolor=fc, edgecolor=ec))

    ax.vlines(snnrs_of_subjects, hist_dat_per_sbuject[np.arange(6, 71, 7)], np.max(np.stack((np.asarray(hist_dat_per_sbuject[np.arange(4, 71, 7)]), np.asarray(hist_dat_per_sbuject[np.arange(5, 71, 7)])), axis=1), axis=1), color='gray', linestyles='dotted', linewidth=3)
    ax.legend(fontsize=15) 
    plt.grid(True) 
    ax.set_ylabel('Relative Score (%)', fontproperties=font) 
    twin_y.set_ylabel('MCC (abs)', fontproperties=font) 
    ax.set_xlabel('Participants SNNR', fontproperties=font) 
    #plt.xticks(ticks=ticks, labels=ticklabels)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    ax.set_ylim([0, 1])
    ax.set_yticks(ticks=ticks)
    ax.set_yticklabels(labels=ticklabels)
    twin_y.set_ylim([0, 1])
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.title('Comparison of ELM-Performance w. Cross-Validated Hyperparameter\n Used participants datasets in modalities [%s]' % (current_title.split('/')[-1]), fontproperties=font)
    plt.tight_layout()
    plt.savefig('/home/christoph/Desktop/Meeting_Bert_22_10/ELM_Plots/%d_%s_plot.png' % (j, current_title.split('/')[-1]))
    plt.close('all')