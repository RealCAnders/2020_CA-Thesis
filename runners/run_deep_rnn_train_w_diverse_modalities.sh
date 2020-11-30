#!/bin/bash

now=$(date '+%d/%m/%Y %H:%M:%S')

python /home/christoph/Desktop/Beginning_October_Work/scripts/deep_rnn_best_param_within_subject_ccar.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/deep_rnn_best_param_within_subject_csp.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/deep_rnn_best_param_within_subject_ccar.py
python /home/christoph/Desktop/Beginning_October_Work/scripts/deep_rnn_best_param_within_subject_csp_hil.py

later=$(date '+%d/%m/%Y %H:%M:%S')
echo "Done computing"
echo "Start-time : $now"
echo "End-time : $later"

# ToDo danach: CCAR_CSP_CSP_HIL-Folder anlegen und danach OwnCNN-Implementations um RNN-Part erweitern