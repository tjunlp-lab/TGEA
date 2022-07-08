#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python MF_GPT.py \
    --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
    --train_file ../data/benchmark6/train.json \
    --validation_file ../data/benchmark6/dev.json \
    --test_file ../data/benchmark6/test.json \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3\
    --per_device_eval_batch_size 8 \
    --output_dir ./tmp/comparision_correct \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
