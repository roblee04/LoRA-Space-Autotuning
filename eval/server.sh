#!/bin/bash
source lm-evaluation-harness/venv/Scripts/activate
python -m llama_cpp.server --model merged_output.gguf --n_gpu_layers 10000