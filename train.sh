#!/bin/bash
python3 dqn_training.py --delay_time 0 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 1 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 2 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 3 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 5 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 10 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 15 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 20 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 25 >> ./training_log/training_1_log.txt
python3 dqn_training.py --delay_time 30 >> ./training_log/training_1_log.txt