#!/bin/bash

now=$(date +"%T")

for i in 0 1 2 3 4 5 6 7 8 9
do
  echo "Looping ... number $i"
  python /home/christoph/Desktop/End_Of_September_Work/prep_dat_final.py $i
done

later=$(date +"%T")
echo "Done preprocessing"
echo "Start-time : $now"
echo "End-time : $later"

#echo "Starting Training ELM"
#now_elm=$(date +"%T")

#python /home/christoph/Desktop/End_Of_September_Work/elm_train_intermediate.py 1
#python /home/christoph/Desktop/End_Of_September_Work/elm_train_intermediate.py 2
#python /home/christoph/Desktop/End_Of_September_Work/elm_train_intermediate.py 3

#later_elm=$(date +"%T")
#echo "Done Training ELM"
#echo "Start-time : $now_elm"
#echo "End-time : $later_elm"
