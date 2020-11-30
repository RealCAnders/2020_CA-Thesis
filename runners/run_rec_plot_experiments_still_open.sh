#!/bin/bash

now=$(date +"%T")

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/DenseConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/rec_plot/ShallowConvNet_on_ccar.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

for k in 1 2 3 4 5 6 7 8 9 10
do
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy' 'k00%d/epoched_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy' 'k00%d/epoched_intrplt_kx_data_combined' $k
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_under_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_100_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy' 'k00%d/epoched_intrplt_filt_over_400_kx' $k
	python /home/christoph/Desktop/Thesis_Plots/Results/2d_tsc/rec_plots/deep_cnn/deep_rnn_rec_plot_ccar_within_subject_aka_deep_cnn.py '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' 'k00%d/epoched_intrplt_filt_500_900_kx' $k
done

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"


# Only open question: (where did the deepRNN run though?!)
# mc-cnn that already ran 
# /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/mc_cnn.py
# /home/christoph/Desktop/Thesis_Plots/Thesis_Code/Thesis_Models/scripts/2d_tsc/mc_classification/eeg_net.py