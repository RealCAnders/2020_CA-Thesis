#!/bin/bash

now=$(date +"%T")

#-#-#for k in 10
#-#-#do
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy' 'outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_out_rej' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy' 'outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_out_rej' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy' 'outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_out_rej' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy' 'outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_out_rej' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy' 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_out_rej' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy' 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy' 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy' 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy' 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx' $k
#-#-#	python /home/christoph/Desktop/Beginning_October_Work/scripts/2d_tsc/mc_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy' 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx' $k
#-#-#done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_intrplt_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_intrplt_filt_under_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_intrplt_filt_over_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_intrplt_filt_over_400_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet.py $k 'epoched_intrplt_filt_500_900_kx'
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_intrplt_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_intrplt_filt_under_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_intrplt_filt_over_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_intrplt_filt_over_400_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet.py $k 'epoched_intrplt_filt_500_900_kx'
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_intrplt_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_intrplt_filt_under_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_intrplt_filt_over_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_intrplt_filt_over_400_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/DenseConvNet_on_ccar.py $k 'epoched_intrplt_filt_500_900_kx'
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_intrplt_kx_data_combined'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_intrplt_filt_under_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_intrplt_filt_over_100_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_intrplt_filt_over_400_kx'
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/ShallowConvNet_on_ccar.py $k 'epoched_intrplt_filt_500_900_kx'
done

for k in 1
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py
done

# echo "Training ELM on data w 1/10 class ratio hfSEP/noise"
# python /home/christoph/Desktop/End_Of_September_Work/elm_train_intermediate_class_imbalances.py

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"
