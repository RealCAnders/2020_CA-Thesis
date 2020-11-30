import math

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 

import numpy as np
from numpy import load

averaging_points = [1, 2, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
ticks_labels = ['1', '2', '5', '10', '20', '50', '100', '200', '500']

font = FontProperties() 
font.set_family('serif') 
font.set_name('Times New Roman') 
font.set_style('oblique') 
font.set_size(20)

optima = [0.181510, 0.392756, 0.573910, 0.090179, 0.234987, 0.186130, 0.201979, 0.293155, 0.201403, 0.326112]
optima_ids = [100, 20, 10, 500, 50, 100, 100, 50, 100, 30]
rel_dist = [50, 2, 1, 200, 5, 50, 50, 5, 50, 3]

for title in[
#	'data_combined',
#	'under_100Hz',
#	'over_100Hz',
#	'over_400Hz',
	'500Hz_900Hz'
]:
	for k in range(1, 11):
		variance_ratios = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_variance_ratios.npy' % (k, title), allow_pickle=True)
		stds_slid = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_stds_slid.npy' % (k, title), allow_pickle=True)
		stds_jump = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_stds_jump.npy' % (k, title), allow_pickle=True)
		snnrs = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_snnrs.npy' % (k, title), allow_pickle=True)
		mean_stds_over_time_slid_hfsep = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_slid_hfsep.npy' % (k, title), allow_pickle=True)
		mean_stds_over_time_slid_baseline = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_slid_baseline.npy' % (k, title), allow_pickle=True)
		std_of_stds_over_time_slid_hfsep = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_slid_hfsep.npy' % (k, title), allow_pickle=True)
		std_of_stds_over_time_slid_baseline = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_slid_baseline.npy' % (k, title), allow_pickle=True)
		mean_stds_over_time_jump_hfsep = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_jump_hfsep.npy' % (k, title), allow_pickle=True)
		mean_stds_over_time_jump_baseline = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_jump_baseline.npy' % (k, title), allow_pickle=True)
		std_of_stds_over_time_jump_hfsep = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_jump_hfsep.npy' % (k, title), allow_pickle=True)
		std_of_stds_over_time_jump_baseline = load('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_jump_baseline.npy' % (k, title), allow_pickle=True)

#-#-#		optimum = 0
#-#-#		optimum_id = 0
#-#-#
#-#-#		for ctr in range(15):
#-#-#			print(math.isclose(std_of_stds_over_time_jump_hfsep[ctr], std_of_stds_over_time_jump_baseline[ctr], rel_tol=0.15))
#-#-#			if math.isclose(std_of_stds_over_time_jump_hfsep[ctr], std_of_stds_over_time_jump_baseline[ctr], rel_tol=0.15):
#-#-#				optimum_id = ctr
#-#-#				optimum = mean_stds_over_time_jump_hfsep[ctr]
#-#-#				print(ctr)
#-#-#				print(optimum)
#-#-#				break
#-#-#
#-#-#		print('#######################')
#-#-#		optimum = 0
#-#-#		optimum_id = 0
#-#-#		previous_value = 0
#-#-#		increase = 0
#-#-#		past_increase = 0
#-#-#		global_lowest_drop = 0
#-#-#
#-#-#		range_of_decrease = mean_stds_over_time_jump_hfsep[:15].ptp()
#-#-#
#-#-#		for count, rms_hfsep in enumerate(mean_stds_over_time_jump_hfsep[:15]):
#-#-#			if count == 0:
#-#-#				increase = 0
#-#-#			else:
#-#-#				increase = rms_hfsep - previous_value
#-#-#				print('%d, %f' % (count, increase))
			#-#-#
#-#-#			if count > 0 and increase > (-0.3 * range_of_decrease):
#-#-#				optimum_id = count
#-#-#				optimum = previous_value
#-#-#				break
#-#-#
#-#-#			previous_value = rms_hfsep
		
#-#		for eind, entry in enumerate(mean_stds_over_time_jump_hfsep[:15]):
#-#			print('%d, %f' % (averaging_points[eind], entry))
#-#
		#-#print('############## %d' % k)

		# ToDo: Move to only jumping windows and limit to 500 as avg_points!
		fig = plt.figure(figsize=(12, 8)) 
		ax = fig.add_subplot(111) 
		lin0 = plt.plot(averaging_points[:15], snnrs[:15,1], linestyle='-', color='#1b9e77', label='mean_snnr', linewidth=2)  
		lin1 = plt.plot(averaging_points[:15], variance_ratios[:15,0], linestyle=':', color='#1b9e77', label='var_rat', linewidth=2)  
		ax.set_ylabel('SNNR, var-rat', fontproperties=font)
		ax.tick_params(axis='y', labelsize=20) 
		ax.tick_params(axis='x', labelsize=20)
		axe = ax.twinx()
		lin2 = axe.plot(averaging_points[:15], mean_stds_over_time_jump_hfsep[:15], linestyle='-', color='#d95f02', label='RMS hfSEP', linewidth=2)  
		lin3 = axe.plot(averaging_points[:15], mean_stds_over_time_jump_baseline[:15], linestyle='-', color='#7570b3', label='RMS baseline', linewidth=2)  
		lin4 = axe.plot(averaging_points[:15], std_of_stds_over_time_jump_hfsep[:15], linestyle=':', color='#d95f02', label='STD of RMS hfSEP', linewidth=2)  
		lin5 = axe.plot(averaging_points[:15], std_of_stds_over_time_jump_baseline[:15], linestyle=':', color='#7570b3', label='STD of RMS baseline', linewidth=2)  
		# plt.title('K00%d [%s]; effect of sub-averaging;' % (k, title), fontproperties=font) 
		axe.set_yscale('log') 
		axe.plot(optima_ids[k-1], optima[k-1], 'ro')
#-#		axe.plot(10, 0.1, 'ro')
		if k == 2:
			axe.annotate('optimum', xy=(optima_ids[k-1], optima[k-1]), xytext=(optima_ids[k-1] - rel_dist[k - 1] - 10, optima[k-1] + 0.1), arrowprops=dict(facecolor='black'), fontsize=15)
		else:
			axe.annotate('optimum', xy=(optima_ids[k-1], optima[k-1]), xytext=(optima_ids[k-1] - rel_dist[k - 1], optima[k-1] + 0.1), arrowprops=dict(facecolor='black'), fontsize=15)
		axe.set_ylim(0.005, 1)
		axe.set_ylabel('RMS, STD of RMS in μV', fontproperties=font)
		axe.tick_params(axis='y', labelsize=20)
		axe.tick_params(axis='x', labelsize=20)
		plt.xscale('log') 
		ax.set_xlabel('# of trials used for sub-averaging', fontproperties=font) 
		lns = lin2 + lin4 + lin3 + lin5 + lin0 + lin1
		lbls = [l.get_label() for l in lns] 
		ax.legend(lns, lbls, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=3)
		plt.subplots_adjust(bottom=0.25)
		plt.xlim([1, 500])
	
		# these are matplotlib.patch.Patch properties
		props = dict(boxstyle='round', facecolor='gray', alpha=0.3)

		names = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006', 'K007', 'K008', 'K009', 'K0010']
		# place a text box in upper left in axes coords
		ax.text(0.45, 0.98, names[k - 1], transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

		ax.set_ylim([0, 30])
		plt.xticks(ticks=ticks, labels=ticks_labels, fontsize=20)
		plt.tick_params(labelsize=20)
		#plt.rcParams.update({'font.size': 20})
		plt.savefig('/home/christoph/Desktop/End_of_november_thesis/Playground_Plots_For_Thesis/snnr_var_analysis_testing_for_displaying/k00%d_plot_%s' % (k, title), dpi=300)
		plt.close('all')

#-#		fig = plt.figure(figsize=(12, 8)) 
#-#		ax = fig.add_subplot(111) 
#-#		lin0 = plt.plot(averaging_points, snnrs[:,0], linestyle='-', color='#1b9e77', label='mean_snnr', linewidth=2)  
#-#		ax.set_ylabel('SNNR', fontproperties=font) 
#-#		ax.set_ylim(0, 90)
#-#		ax2 = plt.twinx() 
#-#		lin2 = ax2.plot(averaging_points, mean_stds_over_time_slid_hfsep, linestyle='-', color='#d95f02', label='mean_stds_hfSEP', linewidth=2)  
#-#		lin3 = ax2.plot(averaging_points, mean_stds_over_time_slid_baseline, linestyle='-', color='#7570b3', label='mean_stds_baseline', linewidth=2)  
#-#		lin1 = ax2.plot(averaging_points, variance_ratios[:,0], linestyle=':', color='#1b9e77', label='var_rat', linewidth=2)  
#-#		lin4 = ax2.plot(averaging_points, std_of_stds_over_time_slid_hfsep, linestyle=':', color='#d95f02', label='std_of_stds_hfSEP', linewidth=2)  
#-#		lin5 = ax2.plot(averaging_points, std_of_stds_over_time_slid_baseline, linestyle=':', color='#7570b3', label='std_of_stds_baseline', linewidth=2)  
#-#		lin6 = ax2.plot(averaging_points, std_of_stds_over_time_slid_hfsep / mean_stds_over_time_slid_hfsep, linestyle='-', color='black', label='coeff_of_var_hfSEP', linewidth=2)
#-#		lin7 = ax2.plot(averaging_points, std_of_stds_over_time_slid_baseline / mean_stds_over_time_slid_baseline, linestyle=':', color='black', label='coeff_of_var_baseline', linewidth=2)
#-#		ax2.set_ylabel('STDs, var-rat and CV', fontproperties=font) 
#-#		plt.title('Variability analysis for K00%d [%s];\nSNNR, standard deviation, variance-ratio and coefficients of variation\nagainst number of trials used for sub-averaging;' % (k, title), fontproperties=font) 
#-#		plt.yscale('log') 
#-#		ax2.set_yscale('log') 
#-#		plt.xscale('log') 
#-#		ax.set_xlabel('# of trials used for sub-averaging', fontproperties=font) 
#-#		lns = lin0 + lin1 + lin2 + lin4 + lin3 + lin5 + lin6 + lin7
#-#		lbls = [l.get_label() for l in lns] 
#-#		ax.legend(lns, lbls, fontsize=15, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
#-#		plt.subplots_adjust(bottom=0.25)
#-#		plt.savefig('/home/christoph/Desktop/Beginning_October_Work/finalest_of_final_of_final_snnr_showcase_most_final_version/k00%d_plot_%s' % (k, title), dpi=400)
#-#		plt.close('all')