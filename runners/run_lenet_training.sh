#!/bin/bash

now=$(date +"%T")

python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_adam_1.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_adam_01.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_adam_05.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_adam_005.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_sgd_1.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_sgd_01.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_sgd_05.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/lenet5_sgd_005.py

later=$(date +"%T")
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"