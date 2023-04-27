#!/bin/bash

export TARGET_FOLDER="/raid/dangerous_man_jinha_chung/LLaMA_weights"
# 7B runs well on A100 (40GB)
#torchrun --nproc_per_node 1 example.py --ckpt_dir $TARGET_FOLDER/7B --tokenizer_path $TARGET_FOLDER/tokenizer.model
# think 13B needs 2 GPUs? error on single A100 (not OOM, CUDA ordinal error)
#torchrun --nproc_per_node 2 example.py --ckpt_dir $TARGET_FOLDER/13B --tokenizer_path $TARGET_FOLDER/tokenizer.model

torchrun --nproc_per_node 1 serve.py --ckpt_dir $TARGET_FOLDER/7B --tokenizer_path $TARGET_FOLDER/tokenizer.model
