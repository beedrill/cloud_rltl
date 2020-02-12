#!/bin/bash
<<<<<<< HEAD
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
#ENV=1
case $key in
    -n|--experiment_name)
    EXPERIMENT="$2"
    shift # past argument
    shift # past value
    ;;
    -f)
    FILE=YES
    shift # past argument
    shift # past value
    ;;
    --env)
    ENV="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

case $ENV in
  0)
  ENV_NAME=TrafficLight-simple-sparse-v0
  ;;
  1)
  ENV_NAME=TrafficLight-simple-medium-v0
  ;;
  2)
  ENV_NAME=TrafficLight-simple-dense-v0
  ;;
  3)
  ENV_NAME=TrafficLight-Lust12408-midnight-v0
  ;;
  4)
  ENV_NAME=TrafficLight-Lust12408-regular-time-v0
  ;;
  5)
  ENV_NAME=TrafficLight-Lust12408-rush-hour-v0
  ;;
  *)    # unknown option
  echo "warning: you didn't specify correct env option"
  ENV_NAME=NULL
  exit 1
  ;;
esac

if [ -z "$EXPERIMENT" ] #check if this parameter is empty
then
    echo "you need to specify experiment name, with -n or --experiment_name"
    exit 1
fi

if [ "$ENV_NAME" == "NULL" ] #check if this parameter is empty
then
    echo "you need to specify correct env option"
    exit 1
fi

echo "runing experiment ${EXPERIMENT}"
echo "env name is ${ENV_NAME}"



delays='0 1 2 3 4 5 6 7 8 9 10'
for delay in $delays
do
    mkdir ./params/${EXPERIMENT}
    touch ./params/${EXPERIMENT}/medium_delay_$delay.txt
    if [ "$FILE" == "YES" ]; then
        python3 dqn_training.py --delay $delay --no_counter --model_saving_path ${EXPERIMENT}/${ENV_NAME}_delay_$delay --env_name ${ENV_NAME} >> ./params/${EXPERIMENT}/medium_delay_$delay.txt
        # execute the script and copy the output to the file
    else
        python3 dqn_training.py --delay $delay --no_counter --model_saving_path ${EXPERIMENT}/${ENV_NAME}_delay_$delay --env_name ${ENV_NAME}
    fi
done

