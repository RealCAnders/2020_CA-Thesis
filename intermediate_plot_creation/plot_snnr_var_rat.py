import matplotlib.pyplot as plt
import numpy as np

from numpy import load

snnrs = load('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k003/snnrs.npy', allow_pickle=True)
stds_jump = load('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k003/stds_jump.npy', allow_pickle=True)
stds_slid = load('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k003/stds_slid.npy', allow_pickle=True)
var_vats = load('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k003/variance_ratios.npy', allow_pickle=True)

averaging_points = [1, 2, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
comparator_labels = ['unfiltered_data_combined', 'under_100Hz', 'over_100Hz', 'over_400Hz', '500Hz_to_900Hz']
comparator_linestyles = ['b*', 'go', 'cv', 'y^', 'k+']

for idx, snnr_level in enumerate(snnrs):
  
  for indeex, snnrs_at_level in enumerate(snnr_level):

    plt.plot(averaging_points, snnrs_at_level, comparator_linestyles[indeex], label='snnrs_' + comparator_labels[indeex])
    plt.plot(averaging_points, snnrs_at_level, comparator_linestyles[indeex], label='snnrs_' + comparator_labels[indeex])
    plt.plot(averaging_points, snnrs_at_level, comparator_linestyles[indeex], label='snnrs_' + comparator_labels[indeex])
    plt.plot(averaging_points, snnrs_at_level, comparator_linestyles[indeex], label='snnrs_' + comparator_labels[indeex])

  plt.title('Variance-in-hfSEP-to-Variance-in-Noise-Ratio K00%i, modalities-comparison\nFormula w. montage CP5-FZ: mean(std(hfSEP-windows) / std(noise-windows))' % (idx + 1), fontproperties=font)
  plt.xlabel('Number of consecutive trials used in sub-averages', fontproperties=font_sides)
  plt.ylabel('Ratio', fontproperties=font_sides)
  plt.ylim([0, 50])
  plt.legend(loc=2, fontsize=10)
  plt.rc('xtick',labelsize=10)
  plt.rc('ytick',labelsize=10)
  plt.xscale('log')
  plt.savefig('/home/christoph/Desktop/End_Of_September_Work/var_rat_plots/k00%d_variance_ratios_under_windowing_effect' % (idx + 1), dpi=300)
  plt.clf()


for idx, modality_level in enumerate(snnrs): 
     
plt.plot(averaging_points, modality_level[:,0], comparator_linestyles[0], label='snnrs_' + comparator_labels[idx]) 
plt.plot(averaging_points, var_vats[idx][:,0], comparator_linestyles[1], label='var_rats_' + comparator_labels[idx]) 
plt.plot(averaging_points, np.asarray([np.mean(stds_slid_per_avg_point) for stds_slid_per_avg_point in stds_slid[0,:,0]]), comparator_linestyles[2], label='mean_stds_hfSEP_' + comparator_labels[idx]) 
plt.plot(averaging_points, np.asarray([np.mean(stds_slid_per_avg_point) for stds_slid_per_avg_point in stds_slid[0,:,1]]), comparator_linestyles[3], label='mean_stds_baseline_' + comparator_labels[idx]) 
plt.legend() 
plt.savefig('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k003/plot_%s' % comparator_labels[idx], dpi=300) 
plt.clf() 
     
