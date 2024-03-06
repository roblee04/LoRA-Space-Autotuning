#!/bin/bash

python="python3.9"

cd ../../
.  run_env_wave_hpc.sh
cd -


cd $GPTUNEROOT/examples/LLM-tuner/
tp=GPTune-Demo
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json

##########################################################################################
################## Illustrate basic functionalities ######################################
tuner=GPTune
rm -rf gptune.db/*.json # do not load any database
#$RUN
python3.9 ./demo-llm.py -optimization ${tuner} -ntask 1 -nrun 5
###########################################################################################



