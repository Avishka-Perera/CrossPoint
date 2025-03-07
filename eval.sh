#! /bin/bash

if [ -z "$1" ]; then
    echo "Error: The state dict path must be specified."
    echo "Usage: $0 <state_dict_path>"
    exit 1
fi
sd_path=$1

echo "Evaluating model with state dict at $sd_path"

echo "Fewshot classification"
python eval_fewshot.py --model_path $sd_path --n_runs 50
