CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --model_name hfl/chinese-macbert-base \
    --train_path ../data/benchmark2/train.json\
    --dev_path ../data/benchmark2/dev.json\
    --test_path ../data/benchmark2/test.json\
    --batch_size_per_gpu 32\
    --gradient_accumulation_steps 2\
    --learning_rate 2e-5 \
    --task_name MiSEWDetection \
    --epochs 3 \
    --do_train True \
    --do_test True \
    --do_eval True \
    --focal_alpha 0.4

