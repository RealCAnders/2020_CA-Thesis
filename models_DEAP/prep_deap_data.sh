#!/bin/bash

now=$(date +"%T")

# Ran already: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
do
  echo "Looping ... number $i"
  python /media/christoph/Volume/Masterthesis/DEAP_TSC/pipeline_DEAP.py $i
done

later=$(date +"%T")
echo "Done preprocessing"
echo "Start-time : $now"
echo "End-time : $later"
