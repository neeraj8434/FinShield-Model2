#!/usr/bin/env python3
"""Precompute frozen LSTM sequence outputs for fast independent head runs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from src.pipeline import load_model


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-checkpoint",required=True);p.add_argument("--cache-dir",required=True)
    p.add_argument("--frames",type=int,default=20);p.add_argument("--num-threads",type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.num_threads)
    base_hash=digest(Path(a.init_checkpoint));root=Path(a.cache_dir)/"features"/base_hash[:16]/f"{a.frames}frames"
    model=load_model(a.init_checkpoint,device="cpu");model.eval();updated=0
    with torch.no_grad():
        for split in ("train","val"):
            for path in sorted((root/split).glob("*.pt")):
                item=torch.load(path,map_location="cpu",weights_only=True)
                if "lstm_outputs" in item:continue
                embeddings=item["embeddings"].float()
                outputs,_=model.lstm(embeddings.unsqueeze(0),None)
                deltas=(embeddings[1:]-embeddings[:-1]).abs().mean(dim=-1)
                if deltas.numel():
                    stats=torch.stack([deltas.mean(),deltas.std(unbiased=False),torch.quantile(deltas,0.9)])
                else:stats=torch.zeros(3)
                item["lstm_outputs"]=outputs[0].cpu().float()
                item["lstm_last"]=outputs[0,-1].cpu().float()
                item["temporal_stats"]=stats.cpu().float()
                torch.save(item,path);updated+=1
                if updated%100==0:print(f"cached frozen LSTM outputs: {updated}",flush=True)
    print(f"Updated {updated} feature files under {root}")


if __name__=="__main__":main()
