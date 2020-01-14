# Deep Q-learning code for cloud-based traffic signal control considering transmission delay

## Citing
To cite this repo for publication:
```
@misc{cloud_dqn,
  author = {Xinze Zhou, Rusheng Zhang},
  title = {Deep Q-learning code for cloud-based traffic signal control considering transmission delay},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/beedrill/cloud_rltl}},
}
```

## TO RUN

```
# training : 
python dqn_training.py --delay_time $DELAY_TIME


# evaluation :
# Used for evaluation for one network and visualization for SUMO
python dqn_evaluation.py --visual

# evaluation2 :
# Used for finding the best network and its average waiting time. 
python dqn_evaluation2.py --delay_time 1 --folder_name "DRL_30"

# train.sh
# Need to specify the train_index and delay_time parameter in train.sh
sh train.sh

# evaluate.sh
# Need to specify the evaluate_index, folder_name and delay_time parameter in evaluate.sh
sh evaluate.sh

```

The trained model's parameters is saved in the folder '/params'

