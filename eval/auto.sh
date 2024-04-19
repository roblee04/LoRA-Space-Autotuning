#!/bin/bash

input_model="../models/llama-2-7b.Q2_K.gguf"
eval_task="lambada_openai"
learning_rate=0.001
data="../data/auxilary_train/science_middle.csv"

# input_model=$1 #"llama-2-7b.Q2_K.gguf"
# eval_task=$2 #"lambada_openai"
# learning_rate=$3 #0.001
# data=$4 #"llama.cpp/build/data/science_middle.csv"

# llama.cpp/build/bin/Release/finetune --model-base llama.cpp/models/$input_model --train-data $data --threads 8 --sample-start "\n" -ngl 10000 --adam-alpha $learning_rate #default 0.001
# llama.cpp/build/bin/Release/export-lora --model-base llama.cpp/models/$input_model --model-out tmp/merged_output.gguf --lora-scaled llama.cpp/ggml-lora-LATEST-f32.gguf 1.0;


../llama.cpp/finetune --model-base $input_model --train-data $data --threads 8 --sample-start "\n" --adam-alpha $learning_rate #default 0.001
../llama.cpp/export-lora --model-base $input_model --model-out merged/merged_output.gguf --lora-scaled lora/ggml-lora-LATEST-f32.gguf 1.0;


sh server.sh &
sleep 15    #this is a bad way of waiting for server to start

#Run Eval
# source lm-evaluation-harness/venv/Scripts/activate
source ../.venv/bin/activate 
lm_eval --model gguf --model_args base_url=http://localhost:8000 --tasks $eval_task --output_path ../output
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT