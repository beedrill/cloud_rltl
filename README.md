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
python dqn_training.py --delay $DELAY_TIME --env_name $ENV_NAME


# evaluation :
# Used for evaluation for one network and visualization for SUMO
python3 dqn_evaluation.py --filename MODELPATH  --delay DELAYTIME --env_name ENV_NAME --n_trials NUMBEROFTRIALS --saving_file SAVINGPATH



# to perform a experiment for multiple delays
# Need to specify the train_index, env_option and delay_time parameter in train.sh
bash train.sh --n SOMENAME ----env 0 

#here are the env options:

                                            env_option
'TrafficLight-simple-sparse-v0'                 0
'TrafficLight-simple-medium-v0'                 1
'TrafficLight-simple-dense-v0'                  2
'TrafficLight-Lust12408-rush-hour-v0'           3
'TrafficLight-Lust12408-regular-time-v0'        4
'TrafficLight-Lust12408-midnight-v0'            5

# To evaluate the experiment
python3 evaluate_experiment.py -p experiment_path


```

The trained model's parameters is saved in the folder '/params'

