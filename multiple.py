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

def readLogIntoList(filename):
    bucket = []
    # read saved file format back into Python list
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tensor2D = line.strip("Sequence #").strip().split(":")
            d = dict()
            d["seqnum"] = int(tensor2D[0])
            tensor2D = tensor2D[1].strip().strip("[[").strip("]]").replace("'", "").split("], [")
            # temp represents a 2D tensor
            for i, tensor1D in enumerate(tensor2D):
                tensor2D[i] = tensor2D[i].split(",")
                for j, elem in enumerate(tensor2D[i]):
                    tensor2D[i][j] = float(elem.strip())
            d["data"] = tensor2D
            bucket.append(d)
    return bucket

def countSparseElements(bucket, threshold = None):
    thresholdFixed = (threshold != None)
    sparseCount = []
    for e in bucket:
        seqnum = e["seqnum"]
        tensor2D = e["data"]
        if not thresholdFixed:
            threshold = 1 / seqnum * 0.1
            #threshold = 1 / seqnum
        mask = [ [1 if v < threshold else 0 for v in r] for r in tensor2D]
        d = dict()
        d["seqnum"] = seqnum
        d["small"] = sum(map(sum, mask))
        d["large"] = len(tensor2D) * len(tensor2D[0]) - d["small"]
        d["sparsity"] = d["small"] / (d["small"] + d["large"])
        sparseCount.append(d)
    return sparseCount          

def getAveragedSparsity(sparseCount):
    sparsitySum = 0.
    for d in sparseCount:
        sparsitySum += d["sparsity"]
    return sparsitySum / len(sparseCount)

def getWeightedSparsity(sparseCount):
    smallSum = 0
    largeSum = 0
    for d in sparseCount:
        smallSum += d["small"]
        largeSum += d["large"]
    return smallSum / (smallSum + largeSum)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    #temperature: float = 0.8,
    temperature: float = 0.9,
    #top_p: float = 0.95,
    top_p: float = 0.8,
    max_seq_len: int = 256,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    score_save_dir = "/home/jinha/LLaMA/llama/scores/"
    score_move_dir = "/home/jinha/LLaMA/llama/scores/multiple/"
    prompt_load_dir = "/home/jinha/LLaMA/llama/prompts/"
    with open(prompt_load_dir + "sparsity_checking.txt", "r") as f:
        rawPrompts = f.readlines()
    presetPrompts = []
    for rp in rawPrompts: 
        presetPrompts.append(rp.strip())

    for i, prompt in enumerate(presetPrompts):
        prompt = [prompt]
        result = generator.generate(prompt, max_gen_len=512, temperature=temperature, top_p=top_p)
        print(f"* ========== Prompt #{format(i, '02')} result:\n{result[0]}")
        # parse log
        bucket = readLogIntoList(score_save_dir + "raw_score.log")
        # count sparse elements
        sparseCount = countSparseElements(bucket)
        # get sparse stats
        averagedSparsity = round(getAveragedSparsity(sparseCount), 4)
        weightedSparsity = round(getWeightedSparsity(sparseCount), 4)
        # rename file
        os.rename(score_save_dir + "raw_score.log", score_move_dir + f"prompt{format(i, '02')}.log")
        #print(sparseCount)
        print(f"* ========== Prompt #{format(i, '02')} Averaged matrix sparsity: {averagedSparsity}")
        print(f"* ========== Prompt #{format(i, '02')} Weighted matrix sparsity: {weightedSparsity}")

    print("* ========== Terminating LLaMA. Good bye! ========== *")

if __name__ == "__main__":
    fire.Fire(main)
