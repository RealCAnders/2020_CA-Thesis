#!/bin/bash

now=$(date +"%T")

#-#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
#-#do
#-#  echo "Looping ... number $i"
#-#  python /media/christoph/Volume/Masterthesis/DEAP_TSC/pipeline_DEAP.py $i
#-#done

# Not possible to run currently, due to issues w. data loading: 23 24 25 26 27 28 29 30 31 32 
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
do
  echo "Looping ... number $i"
  python /media/christoph/Volume/Masterthesis/DEAP_TSC/models/1D/AlexNet.py $i
done

later=$(date +"%T")
echo "Done preprocessing"
echo "Start-time : $now"
echo "End-time : $later"