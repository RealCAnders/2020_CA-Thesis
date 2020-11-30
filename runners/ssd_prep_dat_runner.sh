#!/bin/bash

now=$(date +"%T")

for i in 0 1 2 3 4 5 6 7 8 9
do
  echo "Looping ... number $i"
  python /home/christoph/Desktop/End_Of_September_Work/prep_dat_final_with_SSD.py $i
done

later=$(date +"%T")
echo "Done preprocessing"
echo "Start-time : $now"
echo "End-time : $later"