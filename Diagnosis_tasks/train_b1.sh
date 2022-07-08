CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --model_name hfl/chinese-macbert-base \
    --train_path ../data/benchmark1/train.json\
    --dev_path ../data/benchmark1/dev.json\
    --test_path ../data/benchmark1/test.json\
    --batch_size_per_gpu 48\
    --gradient_accumulation_steps 2\
    --learning_rate 2e-5 \
    --task_name ErroneousDetection \
    --epochs 3 \
    --do_train True \
    --do_eval True \
    --do_test True \
    --focal_alpha 0.3 \
    #--use_focal True
