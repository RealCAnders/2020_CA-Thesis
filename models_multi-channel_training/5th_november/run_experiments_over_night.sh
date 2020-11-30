#!/bin/bash

now=$(date +"%T")

python /media/christoph/Volume/Masterthesis/code_experiments_to_transfer_4th_november/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py

for k in 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Beginning_October_Work/scripts/train_deep_rnn_like_elm_train_intermediate.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"