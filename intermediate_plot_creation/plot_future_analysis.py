save('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/variance_ratios.npy' % (identifier + 1), variance_ratios)
save('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/stds_slid.npy' % (identifier + 1), stds_slid)
save('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/stds_jump.npy' % (identifier + 1), stds_jump)
save('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/snnrs.npy' % (identifier + 1), snnrs)

plt.plot(averaging_points, snnrs[:,0], linestyle='-.', label='snnrs') 
plt.plot(averaging_points, variance_ratios[:,0], linestyle='--', label='var_rats') 
plt.plot(averaging_points, mean_stds_over_time_slid_hfsep, linestyle=':', label='mean_stds_hfSEP') 
plt.plot(averaging_points, mean_stds_over_time_slid_baseline, linestyle='-.', label='mean_stds_baseline') 
plt.plot(averaging_points, std_of_stds_over_time_slid_hfsep, linestyle='--', label='std_of_stds_hfSEP') 
plt.plot(averaging_points, std_of_stds_over_time_slid_baseline, linestyle=':', label='std_of_stds_baseline') 
plt.title('500-900Hz filt, snnr and var-rat showcase;')
plt.xscale('log')
plt.legend() 
plt.savefig('/home/christoph/Desktop/Beginning_October_Work/snnr_var_rat_computed/k00%d/plot_%s' % ((identifier + 1), 'future_analysis_step'), dpi=300) 
plt.clf() 