import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import joblib
from datetime import timedelta
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from trak.projectors import BasicProjector, ProjectionType
from sklearn.decomposition import PCA
from main.data import DATASETS, FileDataset
from main.utils import encode_with_messages_format, encode_prompts, get_train_dataset, get_batch_embeddings

from tqdm import tqdm
import argparse
import os
from accelerate import Accelerator
import torch.distributed as dist
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--save_dir", type=str, default="l")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_dataset", type=str, default="alpaca")
parser.add_argument("--eval_dataset", type=str)
parser.add_argument("--index_path", type=str)
# be careful with this one! leaks test data into train set so we can sanity check the retrieval
parser.add_argument("--leak_test_data", action="store_true")
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--prompt_only", action="store_true")
parser.add_argument("--label_only", action="store_true")
parser.add_argument("--pooling", type=str, default="weighted_mean", choices=["none", "mean", "weighted_mean", "multilayer", "lastlayer"])  # none, mean, weighted_mean
parser.add_argument("--only_first_two", action="store_true")  # only use the first two messages
parser.add_argument("--n_layer", type=int, default=None)
parser.add_argument("--sec_pooling", type=str, default=None, choices=["sum", "cat", "redcat", "avg"])
parser.add_argument("--debug", action="store_true")
parser.add_argument("--trunc_size", type=int, default=2048)
parser.add_argument("--take_layer", type=int, default=-1)
parser.add_argument("--whiten", action="store_true")
parser.add_argument("--reddim", type=int, default=512)
parser.add_argument("--fit", type=int, default=500000)
args = parser.parse_args()

torch.manual_seed(args.seed)
if args.dtype == "bf16":
    kwargs = {"torch_dtype": torch.bfloat16}
elif args.dtype == "fp16":
    kwargs = {"torch_dtype": torch.float16}
elif args.dtype == "fp32":
    kwargs = {"torch_dtype": torch.float32}
if "llama" in args.model_name:
    kwargs["attn_implementation"] = "sdpa"


if os.getenv("HF_TOKEN") is not None:
    kwargs["token"] = os.getenv("HF_TOKEN")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    **kwargs, 
)

model.eval()

if args.tokenizer is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token
# load and process train dataset
if args.train_dataset == "alpaca":
    train_dataset = load_dataset("json", data_files="data/stanford_alpaca_data.jsonl")[
        "train"
    ]
    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(x, tokenizer, 512, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16
    )
elif args.train_dataset == "tulu2":
    train_dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(x, tokenizer, 2048, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16
    )
elif args.train_dataset == "tulu3":
    train_dataset = load_dataset("json", data_files="/tuluv3/tulu3.jsonl", split="train")
    train_dataset = train_dataset.map(
        lambda x: encode_prompts(args.train_dataset, x, tokenizer, args.trunc_size, True, args.label_only, args.only_first_two, args.prompt_only), num_proc=16
    )
elif args.train_dataset == "ling_mini":
    train_dataset = get_train_dataset("/ling_mini_datasets")
    train_dataset= train_dataset.map(
        lambda x: encode_prompts(args.train_dataset, x, tokenizer, args.trunc_size, True, args.label_only, args.only_first_two, args.prompt_only), 
        num_proc=8,
        cache_file_name=f"source/{args.train_dataset}/map_trunc_{args.trunc_size}_model_{os.path.basename(args.model_name)}/ds.arrow"
    )

elif args.train_dataset == "ling2full":
    train_dataset = get_train_dataset("/ling2full")
    train_dataset= train_dataset.map(
        lambda x: encode_prompts(args.train_dataset, x, tokenizer, args.trunc_size, True, args.label_only, args.only_first_two, args.prompt_only), 
        num_proc=8,
        cache_file_name=f"source/{args.train_dataset}/map_trunc_{args.trunc_size}_model_{os.path.basename(args.model_name)}/ds.arrow"
    )
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset("json", data_files=args.train_dataset)["train"]

        def tokenize(x):
            return encode_with_messages_format(x, tokenizer, 2048, True, args.label_only, args.only_first_two, args.prompt_only)
        train_dataset = train_dataset.map(
            tokenize, num_proc=8, load_from_cache_file=True, keep_in_memory=False
        )
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts(
        seed=args.seed, prompt_only=args.prompt_only, response_only=args.label_only, max_length=args.trunc_size
    )
elif os.path.exists(args.eval_dataset):
    # if a file is given, we assume it's a dataset
    test_dataset = FileDataset(args.eval_dataset, tokenizer).get_all_test_prompts(
        prompt_only=args.prompt_only, response_only=args.label_only, max_length=args.trunc_size
    )
else:
    raise ValueError(f"Invalid dataset: {args.eval_dataset}")

if args.leak_test_data:
    # shrink the training data for quicker testing
    train_dataset = train_dataset.select(range(len(test_dataset)))
    # add test data to train data
    for sample in test_dataset:
        train_dataset = train_dataset.add_item({k: v.tolist() for k, v in sample.items()})
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1小时
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False 
)

eval_data_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=args.batch_size, 
    shuffle=False
)

projector = BasicProjector(
                    grad_dim=model.config.hidden_size,
                    proj_dim=model.config.hidden_size // args.n_layer,
                    seed=args.seed,
                    proj_type=ProjectionType.rademacher,
                    device=accelerator.device,
                    dtype=torch.float16,
                    block_size=256,
                )

model, train_data_loader, eval_data_loader = accelerator.prepare(model, train_data_loader, eval_data_loader)

if args.index_path is not None and os.path.exists(args.index_path + f"_{accelerator.process_index}.pt"):
    shard_train_embeds = torch.load(
        args.index_path + f"_{accelerator.process_index}.pt",
        map_location="cpu"
    )

else:
    train_embeds_chunks = []

    for _ , train_inputs in enumerate(tqdm(train_data_loader)):
        train_embeddings = get_batch_embeddings(
            train_inputs, model,
            args.pooling, args.n_layer,
            args.sec_pooling,
            projector,
            args.take_layer
        )

        train_embeds_chunks.append(train_embeddings.detach().cpu())

    shard_train_embeds = torch.cat(train_embeds_chunks, dim=0)

    if not args.index_path:
        args.index_path = "temp/train"

    save_dir = os.path.dirname(args.index_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(shard_train_embeds, args.index_path + f"_{accelerator.process_index}.pt")

accelerator.wait_for_everyone()
if args.whiten:
    if accelerator.is_main_process:
        print("Whitening! (sklearn PCA, save & load)")

    world_size = accelerator.num_processes

    local_n = shard_train_embeds.size(0)

    per_rank = int(args.fit // world_size)
    per_rank = min(per_rank, local_n)
    if per_rank <= 0:
        raise ValueError(f"per_rank={per_rank}, args.fit={args.fit}, local_n={local_n}")

    # random sampling 
    idx = torch.randperm(local_n, device="cpu")[:per_rank]
    rand_chunk = shard_train_embeds.index_select(0, idx).contiguous()  # CPU, dense

    rand_chunk_gpu = rand_chunk.to(accelerator.device, non_blocking=True)

    gathered = accelerator.gather(rand_chunk_gpu)  # CUDA tensor, shape: [per_rank*world, 1, D]

    # fit and save
    if not args.index_path:
        args.index_path = "temp/train"
    save_dir = os.path.dirname(args.index_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    pca_path = args.index_path + f"_whiten_{args.reddim}.pkl"

    if accelerator.is_main_process:
        collect = gathered.squeeze(1).float().cpu().numpy()  # 2D: [N, D]
        whiten = PCA(n_components=args.reddim, whiten=True)
        whiten.fit(collect)
        joblib.dump(whiten, pca_path)

    accelerator.wait_for_everyone()
    whiten = joblib.load(pca_path)

    # transform
    X = shard_train_embeds.squeeze(1).float().cpu().numpy()
    Xw = whiten.transform(X)                                 # [N_local, reddim]

    shard_train_embeds = torch.from_numpy(Xw).unsqueeze(1).to(
        device=accelerator.device, dtype=shard_train_embeds.dtype
    )

print("Computing training embedding norms!")
print(f"train shard shape: {accelerator.process_index}:", shard_train_embeds.shape)
norms = shard_train_embeds.norm(p=2, dim=2, keepdim=True).clamp_min_(1e-12)
shard_train_embeds.div_(norms)
shard_train_embeds = shard_train_embeds.squeeze(1)

accelerator.wait_for_everyone()

print("Computing test embeddings!")

if accelerator.is_main_process:
    test_embeds_chunks = []
else:
    test_embeds_chunks = None

for index, test_inputs in enumerate(tqdm(eval_data_loader)):
    test_embeddings = get_batch_embeddings(
        test_inputs, model,
        args.pooling, args.n_layer,
        args.sec_pooling,
        projector,
        args.take_layer
    )
    if args.whiten:
        X = test_embeddings.squeeze(1).float().cpu().numpy()
        Xw = whiten.transform(X)
        test_embeddings = torch.from_numpy(Xw).unsqueeze(1).to(device=accelerator.device, dtype=test_embeddings.dtype)

    test_embeddings = F.normalize(test_embeddings, p=2, dim=-1).squeeze(1)

    test_gathered = accelerator.gather_for_metrics(test_embeddings)

    if accelerator.is_main_process:
        test_embeds_chunks.append(test_gathered.cpu())

all_test_embeds = None

if accelerator.is_main_process:
    all_test_embeds = (
        torch.cat(test_embeds_chunks, dim=0)
        if test_embeds_chunks else None
    )

accelerator.wait_for_everyone()

if dist.is_initialized():
    obj_list_test = [all_test_embeds] if accelerator.is_main_process else [None]
    dist.broadcast_object_list(obj_list_test, src=0)
    all_test_embeds = obj_list_test[0].to(accelerator.device)

else:
    all_test_embeds = all_test_embeds.to(accelerator.device)
print(f"test full shape: {accelerator.process_index}:", all_test_embeds.shape)

shard_train_embeds = shard_train_embeds.to(accelerator.device, dtype=torch.float32, non_blocking=True)
all_test_embeds = all_test_embeds.to(dtype=torch.float32, non_blocking=True)

sim_local = shard_train_embeds @ all_test_embeds.T

print(f"sim shard shape: {accelerator.process_index}:", sim_local.shape)

if dist.is_initialized():
    sim_full = accelerator.gather_for_metrics(sim_local)  # [train_all, test_all]
else:
    sim_full = sim_local
    
if accelerator.is_main_process:
    world_size = accelerator.num_processes

    n_total, n_test = sim_full.shape
    assert n_total % world_size == 0, f"n_total={n_total} not divisible by world_size={world_size}"
    per_rank = n_total // world_size

    sim_ordered = (
        sim_full
        .view(world_size, per_rank, n_test)
        .transpose(0, 1)
        .reshape(n_total, n_test)
    )
    sim_influences = sim_ordered[:len(train_dataset)].T.contiguous().detach().cpu()

    print("Final sim shape:", sim_influences.shape)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.prompt_only:
        save_name = f"{args.eval_dataset}_cossim_promptonly.pt"
        torch.save(sim_influences, os.path.join(args.save_dir, save_name))

    elif args.label_only:
        save_name = f"{args.eval_dataset}_cossim_labelonly.pt"
        torch.save(sim_influences, os.path.join(args.save_dir, save_name))

    else:
        basename = os.path.basename(args.eval_dataset)
        save_to = f"{basename}_{args.n_layer}_{args.sec_pooling}_{os.path.basename(args.model_name)}_{args.trunc_size}"
        if args.whiten:
            save_to = save_to + f"_whiten_reddim_{args.reddim}"

        torch.save(sim_influences, os.path.join(args.save_dir, save_to + "_cossim.pt"))
        print("saved", os.path.join(args.save_dir, save_to + "_cossim.pt"))
