python -m accelerate.commands.launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/llama2-7b-hf \
        --seed 42 \
        --train_dataset ling2full \
        --eval_dataset gsm8k_shots \
        --index_path ling2/llama2_7b_2048_lastlayer_1/train \
        --save_dir ling2/llama2_7b_2048_lastlayer_1/ \
        --batch_size 1 \
        --pooling lastlayer \
        --n_layer 1 \
        --sec_pooling redcat \
        --trunc_size 2048 \
        --whiten \
        --reddim 512 \
        --fit 500000

or

accelerate launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/llama2-7b-hf \
        --seed 42 \
        --train_dataset ling2full \
        --eval_dataset gsm8k_shots \
        --index_path ling2/llama2_7b_2048_lastlayer_1/train \
        --save_dir ling2/llama2_7b_2048_lastlayer_1/ \
        --batch_size 1 \
        --pooling lastlayer \
        --n_layer 1 \
        --sec_pooling redcat \
        --trunc_size 2048 \
        --whiten \
        --reddim 512 \
        --fit 500000