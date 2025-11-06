import argparse
import json
from collections import defaultdict
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from model import GPTConfig, GPT

args = argparse.ArgumentParser()
args.add_argument('--init', type=str, required=True)
args.add_argument('--bias', action='store_true')
args = args.parse_args()

rootd = "data"
rootm = "/project/home/p200535/project/nanogpt-mup"

seed = 42
pl.seed_everything(seed)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
ds = 'shakespeare_char'
data_dir = os.path.join(rootd, ds)
block_size = 1024
batch_size = 1
eval_iters = 10

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_batch(split):
    if split == 'tr':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['tr', 'vl']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    return out

def interpolate_weights(w1, w2, n, return_alphas=True):
    weights = []
    alphas = np.linspace(0, 1, n + 2)
    for alpha in alphas:
        w = {k: (1 - alpha) * w1[k] + alpha * w2[k] for k in w1.keys()}
        weights.append(w)

    if return_alphas:
        return alphas, weights
    return weights

seeds = [1, 2, 3]
init = args.init
lrs = [
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
    0.00048828125,
    0.000244140625,
    0.0001220703125,
    0.00006103515625,
    0.00003051757812,
    0.00001525878906,
    0.000007629394531,
    0.000003814697266,
]
# widths, depths = [128, 512, 2048, 8192], [2]
# depth = depths[0]
widths, depths = [256], [2, 8, 32, 128]
width = widths[0]

depth_alpha_enabled = args.init == "completep"
mup_enabled = args.init == "mup" or depth_alpha_enabled

print("mup enabled: ", mup_enabled)
print("depth enabled: ", depth_alpha_enabled)
print(init)
print(widths)

summary = {}
for s in seeds:
    summary[s] = {}

    for lr in tqdm(lrs):
        summary[s][lr] = {}

        for depth in depths:
            name = f'width{width}_depth{depth}_seed1_lr{lr:.20f}'.rstrip('0')
            path = os.path.join(rootm, ds, init, name)
            if not os.path.isdir(path):
                print(f"missing: {path}")
                continue

            head_size = 64
            n_heads = width // head_size
            model_args = dict(
                n_layer=depth,
                n_head=n_heads,
                n_embd=width,
                block_size=block_size,
                bias=args.bias,
                vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
                dropout=0.0,
                mup_enabled=mup_enabled,
                mup_input_alpha=1.0,
                mup_output_alpha=1.0,
                mup_width_multiplier=width / widths[0] if mup_enabled else 1.0,
                depth_alpha_enabled=depth_alpha_enabled,
                depth_alpha_exp=1.0,
                depth_multiplier=depth / depths[0] if depth_alpha_enabled else 1.0,
            )
            gptconf = GPTConfig(**model_args)
            
            # Merge weights
            name1 = f"width{width}_depth{depth}_seed{s}_lr{lr:.20f}".rstrip('0')
            name2 = f"width{width}_depth{depth}_seed{s%3+1}_lr{lr:.20f}".rstrip('0')
            state_dict1 = torch.load(
                f"{rootm}/{ds}/{init}/{name1}/ckpt_last.pt",
                weights_only=True,
                map_location='cpu',  # load on CPU to save GPU memory
            )["model"]
            state_dict2 = torch.load(
                f"{rootm}/{ds}/{init}/{name2}/ckpt_last.pt",
                weights_only=True,
                map_location='cpu',
            )["model"]
            alphas, weights = interpolate_weights(state_dict1, state_dict2, n=9)
            
            # Evaluation
            m = GPT(gptconf).to(DEVICE)
            summary[s][lr][depth] = defaultdict(list)
            for i, w in enumerate(weights):
                m.load_state_dict(w)
                losses = estimate_loss(m)
                summary[s][lr][depth]["tr_loss"].append(losses['tr'])
                summary[s][lr][depth]["vl_loss"].append(losses['vl'])
                print(f"i {i} lr {lr} tr {losses['tr']:.4f} vl {losses['vl']:.4f}")


with open(f"results/shakespeare_char/merge_{init}.json", "w") as f:
    json.dump(summary, f)
