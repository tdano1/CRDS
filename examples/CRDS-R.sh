python -m accelerate.commands.launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/Qwen3-8B \
        --seed 42 \
        --train_dataset ling2full \
        --eval_dataset gsm8k_shots \
        --index_path ling2/qwen38b_2048_lastlayer_18/train \
        --save_dir ling2/qwen38b_2048_lastlayer_18/ \
        --batch_size 1 \
        --pooling lastlayer \
        --n_layer 18 \
        --sec_pooling redcat \
        --trunc_size 2048

or 

accelerate launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/Qwen3-8B \
        --seed 42 \
        --train_dataset ling2full \
        --eval_dataset gsm8k_shots \
        --index_path ling2/qwen38b_2048_lastlayer_18/train \
        --save_dir ling2/qwen38b_2048_lastlayer_18/ \
        --batch_size 1 \
        --pooling lastlayer \
        --n_layer 18 \
        --sec_pooling redcat \
        --trunc_size 2048