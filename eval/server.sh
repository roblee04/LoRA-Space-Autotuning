#!/bin/bash
# source lm-evaluation-harness/venv/Scripts/activate

source ../.venv/bin/activate 

python -m llama_cpp.server --model ../models/merged_output.gguf
# python -m llama_cpp.server --model ../models/llama-2-7b.Q2_K.gguf --n_gpu_layers 10000