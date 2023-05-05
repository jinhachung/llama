from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    llama_size = None
    for n in ["7B", "13B", "33B", "65B"]:
        if n in str(ckpt_path):
            llama_size = n
    assert llama_size != None
    print(f"* ========== Loaded LLaMA-{llama_size} in {time.time() - start_time:.2f} seconds ========== *")
    print("* ========= [LLaMA config] ========== *")
    print(f"max_seq_len: {model_args.max_seq_len}, max_batch_size: {model_args.max_batch_size}, dim: {model_args.dim}, n_layers: {model_args.n_layers}, n_heads: {model_args.n_heads}, vocab_size: {model_args.vocab_size}, local_rank: {local_rank}, world_size: {world_size}")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 256,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    print("* ========== LLaMA is up and ready to go! ========== *")
    while (True):
        prompt = input("* ========== Type in an input prompt and press enter to generate. Type q to quit. ========== *\n")
        if prompt == "q":
            break
        prompts = [prompt]
        results = generator.generate(prompts, max_gen_len=512, temperature=temperature, top_p=top_p)
        print(results[0])
        
    print("* ========== Terminating LLaMA. Good bye! ========== *")

if __name__ == "__main__":
    fire.Fire(main)
