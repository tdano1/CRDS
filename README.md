#  Compressed Representation Data Selection (CRDS)

## Introduction
This is the source code for the paper [On Representation Redundancy in Large-Scale Instruction Tuning Data Selection](https://arxiv.org/pdf/2602.13773). This repository provides efficient implementations of the two algorithms introduced in the paper—**CRDS-R** (based on Rademacher projection-concatenation) and **CRDS-W** (based on Whitening-based dimensionality reduction)—within a distributed computing framework. This project is built upon the [RDS+](https://github.com/hamishivi/automated-instruction-selection) repository.

## Preparation

### 1. Model Installation
To start this project, first we need to install models (encoders) and datasets that we need to use. For example, the encoders we use are usually installed in the file `/model`, download through Huggingface.

### 2. Data Pool Installation

Dataset installation procedures vary depending on the specific datasets required for your use case.

#### General Users (Open-Source Datasets)
For those using standard open-source data, we provide a streamlined setup. The **tulu-3-sft-mixture** is pre-configured in the codebase. To begin:

(1) Download the dataset from [Hugging Face](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture).

(2) Save the files to your local directory at `/tuluv3`.

#### Enterprise Users (Custom Datasets)
Enterprise users utilizing proprietary datasets must perform additional configuration steps to ensure compatibility with the pipeline:

(1) **Dataset Declaration**: In `compute_influence_cosinesim_parallel.py`, register your custom datasets within the `args.datasets` argument.

(2) **Preprocessing**: In `utils.py`, implement the necessary preprocessing logic. This step transforms raw text into a standardized format optimized for representation computing and similarity comparisons.

(3) **Final Integration**: Once steps (1) and (2) are complete, the forward computing pipeline is ready. To finalize dataset generation, declare your datasets in `get_top_influences.py`. 

> **Note:** The implementation in `get_top_influences.py` generally mirrors the logic used in step (1).

### 3. Test Dataset Installation
This repository includes the **GSM8k** and **MBPP** datasets for similarity computation. To use them, set the `--eval_dataset` flag to `gsm8k_shots` or `mbpp_pro`.

If additional datasets (such as **MMLU** or **BBH**) are required, run the following command to install the evaluation data:

```bash
bash download_eval_data.sh
```

Note: Under default settings, the `eval` file is installed in the root directory `/` to ensure compatibility with GPU clusters.

### 4. Dependencies
The environment is set up using Anaconda. Run the following commands to initialize it:

```bash
conda create -n CRDS python=3.10
conda activate CRDS
pip install -r requirements.txt
```

### Similarity Computation
After preparation, start the similarity computation. While CRDS-R and CRDS-W use different initialization scripts, both require updating the training dataset parameters based on the actual data.

#### CRDS-R
```bash
python -m accelerate.commands.launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/Qwen3-8B \
        --seed 42 \
        --train_dataset ${train_dataset} \
        --eval_dataset gsm8k_shots \
        --index_path ling2/qwen38b_2048_lastlayer_18/train \
        --save_dir ling2/qwen38b_2048_lastlayer_18/ \
        --batch_size 1 \
        --pooling lastlayer \
        --n_layer 18 \
        --sec_pooling redcat \
        --trunc_size 2048
```
#### CRDS-W
```bash
python -m accelerate.commands.launch --num_processes 8 \
        -m main.compute_influence_cosinesim_parallel \
        --model_name /models/llama2-7b-hf \
        --seed 42 \
        --train_dataset ${train_dataset} \
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
```
#### For Multi-testset 
Once the representation of a data pool has been computed, it can be efficiently reused across multiple test datasets—a key advantage of this framework. In practice, generating representations for a pool of 2.0M samples typically requires 3–4 hours, whereas computing representations for test datasets and evaluating their similarities with the precomputed pool incurs significantly lower computational overhead. Below is a reference script used for computation of representation similarity in our paper:
```bash
for dataset in gsm8k_shots mmlu_shots bbh_shots mbpp_pro; do
    python -m accelerate.commands.launch --num_processes 8 \
            -m main.compute_influence_cosinesim_parallel \
            --model_name /ossfs/workspace/models/models/Qwen3-8B \
            --seed 42 \
            --train_dataset ling2full \
            --eval_dataset ${dataset} \
            --index_path ling2full/qwen3_8b_2048_lastlayer_18/train \
            --save_dir ling2full/qwen3_8b_2048_lastlayer_18 \
            --batch_size 1 \
            --pooling lastlayer \
            --n_layer 18 \
            --sec_pooling redcat \
            --trunc_size 2048
done
```
### Dataset Generation
Once the forward computation is complete, run the following script to generate the dataset. Ensure you update `output_size` and `train_dataset` to match your requirements, and point `input_files` to the computed similarity file.

```bash
python -m main.get_top_influences \
        --input_files ling2/qwen38b_2048_lastlayer_18/gsm8k_shots_18_redcat_Qwen3-8B_2048_cossim.pt \
        --output_file ling2/qwen38b_2048_lastlayer_18/test_dataset.json \
        --output_size ${output_size} --selection_method max \
        --train_datasets  ${train_dataset} \
        --output_dataset
```
The resulting dataset is then ready for immediate use in model training.

### Citation
This work can be cited as:
```bash
@article{shu2026representation,
  title={On Representation Redundancy in Large-Scale Instruction Tuning Data Selection},
  author={Shu, Youwei and Zheng, Shaomian and Jin, Dingnan and Qu, Wenjie and Guo, Ziyao and Cui, Qing and Zhou, Jun and Zhang, Jiaheng},
  journal={arXiv preprint arXiv:2602.13773},
  year={2026}
}
```

### Contact
For any further questions or discussion, please feel free to reach out Youwei at shuyw001@gmail.com and/or e1374504@u.nus.edu.