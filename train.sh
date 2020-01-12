#!/bin/bash
PYTHON=`which python3`
for i in {1,2,5,10}
do
   PYTHON dqn_training.py --delay_time $i >> delay_time_$i_log.txt
done