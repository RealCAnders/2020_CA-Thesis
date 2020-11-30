#!/bin/bash

now=$(date +"%T")

for i in 5 3 1
do
    for j in 1.0 0.5 0.1 0.005
    do
        for k in 3 5 7
        do
            for l in 'tanh' 'relu'
            do
                for m in 16 32 64
                do
                	python /home/christoph/Desktop/Beginning_October_Work/scripts/simple_model_after_lenet.py $i $j $k $l $m
                	python /home/christoph/Desktop/Beginning_October_Work/scripts/complex_model_after_AlexNet.py $i $j $k $l $m
                done
            done
        done
    done
done

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"
