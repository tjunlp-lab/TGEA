#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bart_model.py \
    --model_name_or_path fnlp/bart-base-chinese \
    --train_file ../data/benchmark5/train.json\
    --validation_file ../data/benchmark5/dev.json\
    --test_file ../data/benchmark5/test.json\
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir ./tmp/result_v2\
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=12 \
    --overwrite_output_dir \
    --predict_with_generate \
