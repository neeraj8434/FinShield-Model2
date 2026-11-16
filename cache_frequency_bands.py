#!/usr/bin/env python3
"""Add normalized low/mid/high DCT features to the frozen feature cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import torch

from src.frequency_feature import LightweightFrequencyFeature


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--data-root",required=True)
    p.add_argument("--init-checkpoint",required=True);p.add_argument("--cache-dir",required=True)
    p.add_argument("--frames",type=int,default=20);p.add_argument("--face-size",type=int,default=112)
    a=p.parse_args();manifest=Path(a.manifest);root=Path(a.data_root);cache=Path(a.cache_dir)
    initial_hash=file_hash(Path(a.init_checkpoint));features=cache/"features"/initial_hash[:16]/f"{a.frames}frames"
    detector=LightweightFrequencyFeature().eval();updated=0;missing=0
    with manifest.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split=row.get("split","").strip().lower()
            if split not in {"train","val"}:continue
            video=Path(row["video_path"].strip())
            if not video.is_absolute():video=root/video
            video=video.resolve()
            if not video.is_file():missing+=1;continue
            fingerprint=f"{video}|{video.stat().st_size}|{video.stat().st_mtime_ns}|{a.frames}|{a.face_size}"
            feature_key=hashlib.sha256(f"{fingerprint}|{initial_hash}".encode()).hexdigest()
            feature_path=features/split/f"{feature_key}.pt"
            if not feature_path.is_file():continue
            item=torch.load(feature_path,map_location="cpu",weights_only=True)
            if "frequency_bands" in item:continue
            crop_key=hashlib.sha256(fingerprint.encode()).hexdigest();crop_path=cache/f"{crop_key}.pt"
            if not crop_path.is_file():missing+=1;continue
            crops=torch.load(crop_path,map_location="cpu",weights_only=True)["frames"].float()
            if crops.shape[0]<2:continue
            with torch.no_grad():item["frequency_bands"]=detector.band_energies(crops.unsqueeze(0))[0].cpu()
            torch.save(item,feature_path);updated+=1
            if updated%100==0:print(f"cached DCT bands: {updated}",flush=True)
    print(f"Updated feature rows: {updated}; missing source/cache rows: {missing}; feature root: {features}")


if __name__=="__main__":main()
