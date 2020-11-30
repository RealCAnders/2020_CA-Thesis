#!/bin/bash

#ToDo: store python objects for both and create figures from these.
now=$(date +"%T")

# create recurrence plots for thesis - bipolar montages FZ-CP5; plus CCAr for med. expert. suggestion
#-#for i in 1 2
#-#do
#-#  echo "Looping ... number $i"
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_wideband.py $i
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_le_100.py $i
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_ge_100.py $i
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_ge_400.py $i
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_500_900.py $i
#-#  python /home/christoph/Desktop/Thesis_Plots/recplot_scripting_500_900_ccar.py $i
#-#done

# create spectrograms for thesis - bipolar montages FZ-CP5; plus CCAr for med. expert. suggestion
for i in 2 1
do
  echo "Looping ... number $i"
  python /home/christoph/Desktop/Thesis_Plots/specs_wideband.py $i
#  python /home/christoph/Desktop/Thesis_Plots/specs_le_100.py $i
#  python /home/christoph/Desktop/Thesis_Plots/specs_ge_100.py $i
  python /home/christoph/Desktop/Thesis_Plots/specs_ge_400.py $i
#  python /home/christoph/Desktop/Thesis_Plots/specs_500_900.py $i
  python /home/christoph/Desktop/Thesis_Plots/specs_500_900_ccar.py $i
done

later=$(date +"%T")
echo "Done preprocessing"
echo "Start-time : $now"
echo "End-time : $later"