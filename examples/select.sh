python -m main.get_top_influences \
        --input_files ling2/qwen38b_2048_lastlayer_18/gsm8k_shots_18_redcat_Qwen3-8B_2048_cossim.pt \
        --output_file ling2/qwen38b_2048_lastlayer_18/gsm8k_shots_18_redcat_Qwen3-8B_2048_cossim_70k.json \
        --output_size 70000 --selection_method max \
        --train_datasets  ling2full \
        --output_dataset

python -m main.get_top_influences \
        --input_files ling2/llama2_7b_2048_lastlayer_1/gsm8k_shots_1_redcat_llama2-7b-hf_2048_cossim.pt \
        --output_file ling2/llama2_7b_2048_lastlayer_1/gsm8k_shots_1_redcat_llama2-7b-hf_2048_cossim_70k.json \
        --output_size 70000 --selection_method max \
        --train_datasets  ling2full \
        --output_dataset