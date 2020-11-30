#!/bin/bash

now=$(date +"%T")

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/eeg_net.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1
do
	python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/multi-channel_training/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py
done

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"
