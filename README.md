# LoRA-Space-Autotuning
We are going to tune the LoRA hyperparameter space with an autotuner.

# TODOS
- QLoRA training. Send output to ./lora

## Usage
Run through GPTune. Adjust scripts in eval for gpu offloading.

```sh run_examples_wave_hpc.sh```

## Requirements
install gptune and requirements
```git clone https://github.com/gptune/GPTune.git```

install llama.cpp and requirements
```git clone https://github.com/ggerganov/llama.cpp.git```

setup openai endpoint \
https://github.com/EleutherAI/lm-evaluation-harness/issues/1254

install lm-evaluation-harness and requirements
```git clone https://github.com/EleutherAI/lm-evaluation-harness.git```

install / load CUDA 

## Dataset

https://huggingface.co/datasets/EleutherAI/lambada_openai

## Model
https://huggingface.co/TheBloke/Llama-2-7B-GGUF

## Useful snippets

**Models in lora, merged are placeholders for testing. Dataset is lambada.**

**HPC**
srun --partition=gpu --nodes 1 --ntasks 1 --cpus-per-task 4 --mem=32G --gres=gpu:1 --time 0-02:00:00 --pty /bin/bash

srun --partition=amd --nodes 1 --ntasks 1 --cpus-per-task 8 --mem=64G --gres=gpu:1 --time 0-02:00:00 --pty /bin/bash

**llama.cpp** 

build/bin/finetune --model-base models/llama-2-7b.Q2_K.gguf --train-data build/data/test.txt --threads 4 --sample-start "<s>"

build/bin/main -m models/llama-2-7b.Q2_K.gguf -n 128 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt

build/bin/Release/finetune --model-base models/llama-2-7b.Q2_K.gguf --train-data build/data/science_middle.csv --threads 8 --sample-start "\n"

