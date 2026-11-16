#!/usr/bin/env python3
"""Train independent temporal-only and DCT-only fusion heads from cached features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.pipeline import load_model

LABELS = {"real": 0, "0": 0, "fake": 1, "1": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        item = torch.load(self.files[index], map_location="cpu", weights_only=True)
        return item["lstm_last"], item["temporal"], item["frequency"], item["label"]


def collect(manifest: Path, root: Path, base: Path, frames: int, face_size: int, initial_hash: str):
    found = {"train": [], "val": []}
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower(); label = LABELS.get(row.get("label", "").strip().lower())
            if split not in found or label is None: continue
            video = Path(row["video_path"].strip())
            if not video.is_absolute(): video = root / video
            video = video.resolve()
            if not video.is_file(): continue
            key = hashlib.sha256(f"{video}|{video.stat().st_size}|{video.stat().st_mtime_ns}|{frames}|{face_size}|{initial_hash}".encode()).hexdigest()
            feature = base / split / f"{key}.pt"
            if feature.is_file(): found[split].append(feature)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--init-checkpoint", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--frames", type=int, default=20)
    p.add_argument("--face-size", type=int, default=112); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--only-variant", choices=["temporal_only", "dct_only"], default=None,
                   help="Train one independent head, useful for resuming/repeating a controlled run")
    a = p.parse_args(); torch.manual_seed(a.seed)
    torch.set_num_threads(a.num_threads)
    manifest=Path(a.manifest); initial=Path(a.init_checkpoint); initial_hash=sha256(initial)
    base=Path(a.cache_dir)/"features"/initial_hash[:16]/f"{a.frames}frames"
    files=collect(manifest,Path(a.data_root),base,a.frames,a.face_size,initial_hash)
    datasets={split:FeatureDataset(items) for split,items in files.items()}
    counts={split:Counter(int(torch.load(path,map_location="cpu",weights_only=True)["label"]) for path in ds.files)
            for split,ds in datasets.items()}
    for split in ("train","val"):
        if not counts[split][0] or not counts[split][1]:
            raise RuntimeError(f"Missing cached REAL/FAKE {split} features: {dict(counts[split])}")
    outdir=Path(a.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    variants=(a.only_variant,) if a.only_variant else ("temporal_only","dct_only")
    for variant in variants:
        torch.manual_seed(a.seed)
        model=load_model(initial,device="cpu");model.inference_variant=variant
        for name,param in model.named_parameters(): param.requires_grad=name.startswith(("fusion_fc1","fusion_fc2"))
        optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr)
        weights=torch.tensor([sum(counts['train'].values())/(2*counts['train'][i]) for i in (0,1)])
        criterion=nn.CrossEntropyLoss(weight=weights)
        loaders={s:DataLoader(ds,batch_size=1,shuffle=(s=="train")) for s,ds in datasets.items()}
        best=-1.;output=outdir/f"{variant}.pt";log_path=output.with_suffix('.jsonl')
        config={"variant":variant,"manifest":str(manifest.resolve()),"manifest_sha256":sha256(manifest),
                "initial_checkpoint_sha256":initial_hash,"trainable_layers":["fusion_fc1","fusion_fc2"],
                "loss":"weighted CrossEntropyLoss","optimizer":"AdamW","learning_rate":a.lr,
                "frames":a.frames,"face_size":a.face_size,"epochs":a.epochs,"seed":a.seed,"num_threads":a.num_threads,
                "class_counts":{"train":dict(counts['train']),"val":dict(counts['val'])}}
        output.with_suffix('.config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        with log_path.open('w',encoding='utf-8') as log:
            for epoch in range(1,a.epochs+1):
                model.train();train_losses=[]
                for summary,temporal,frequency,labels in loaders['train']:
                    optimizer.zero_grad(set_to_none=True)
                    loss=criterion(model.classify_fused_summary(summary,temporal,frequency),labels)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    loss.backward()
                    if model.fusion_fc1.weight.grad is None:raise RuntimeError('Fusion head got no gradient')
                    optimizer.step();train_losses.append(float(loss.detach()))
                model.eval();cm=torch.zeros(2,2,dtype=torch.long);val_losses=[]
                with torch.no_grad():
                    for summary,temporal,frequency,labels in loaders['val']:
                        logits=model.classify_fused_summary(summary,temporal,frequency)
                        val_losses.append(float(criterion(logits,labels)))
                        for y,pred in zip(labels.tolist(),logits.argmax(-1).tolist()):cm[y,pred]+=1
                tn,fp,fn,tp=cm.flatten().tolist();f1=[]
                for tp0,fp0,fn0 in ((tn,fn,fp),(tp,fp,fn)):
                    precision=tp0/(tp0+fp0) if tp0+fp0 else 0.;recall=tp0/(tp0+fn0) if tp0+fn0 else 0.
                    f1.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
                macro=sum(f1)/2
                record={"epoch":epoch,"train_loss":sum(train_losses)/len(train_losses),"val_loss":sum(val_losses)/len(val_losses),
                        "val_macro_f1":macro,"val_confusion_matrix":cm.tolist(),"train_count":len(datasets['train']),
                        "val_count":len(datasets['val']),"variant":variant,"seed":a.seed,"lr":a.lr}
                log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if macro>best:
                    best=macro
                    torch.save({"state_dict":model.state_dict(),"variant":variant,"label_mapping":{"REAL":0,"FAKE":1},
                                "frames":a.frames,"face_size":a.face_size,"normalization":"ImageNet mean/std",
                                "seed":a.seed,"best_val_macro_f1":best,"initial_checkpoint_sha256":initial_hash},output)
        print(f"{variant}: best validation macro F1={best:.4f}; checkpoint={output}",flush=True)


if __name__ == '__main__':main()
