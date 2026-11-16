#!/usr/bin/env python3
"""Re-encode, resize, and lower-frame-rate a fixed labelled video set."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2


def make_variant(source: Path, target: Path, mode: str) -> dict:
    capture=cv2.VideoCapture(str(source))
    if not capture.isOpened(): raise RuntimeError(f"Cannot open source video: {source}")
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w,out_h=width,height
    output_fps=fps
    if mode=="resize": out_w=max(64,width//2);out_h=max(64,height//2)
    elif mode=="fps_half": output_fps=max(1.0,fps/2.0)
    target.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*"mp4v"),output_fps,(out_w,out_h))
    if not writer.isOpened(): capture.release();raise RuntimeError(f"Cannot create video: {target}")
    read=written=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            current=read;read+=1
            if mode=="fps_half" and current%2:continue
            if mode=="resize":frame=cv2.resize(frame,(out_w,out_h),interpolation=cv2.INTER_AREA)
            writer.write(frame);written+=1
    finally:
        capture.release();writer.release()
    if written==0:raise RuntimeError(f"No frames written for {source}")
    return {"source":str(source),"output":str(target),"mode":mode,"source_frames":frame_count,
            "output_frames":written,"source_fps":fps,"output_fps":output_fps,
            "source_resolution":[width,height],"output_resolution":[out_w,out_h]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--data-root",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--reference-json",required=True);p.add_argument("--frames",type=int,default=20)
    p.add_argument("--device",default="cpu");p.add_argument("--split",default="test")
    a=p.parse_args();manifest=Path(a.manifest);root=Path(a.data_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    with manifest.open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r.get("split","").strip().lower()==a.split.lower()]
    if not rows:raise RuntimeError(f"No rows with split={a.split!r}")
    reference=json.loads(Path(a.reference_json).read_text(encoding="utf-8"))
    modes=("reencode","resize","fps_half");transforms=[];comparisons={}
    for mode in modes:
        variant_rows=[];metadata=[]
        for row in rows:
            source=Path(row["video_path"].strip())
            if not source.is_absolute():source=root/source
            target=out/mode/f"{source.stem}.mp4"
            info=make_variant(source,target,mode);metadata.append(info)
            variant_rows.append({"video_path":str(target.resolve()),"label":row["label"],"split":"test",
                                 "source_id":row.get("source_id",""),"group_id":row.get("group_id",""),
                                 "transformation":mode})
        variant_manifest=out/f"{mode}.csv"
        with variant_manifest.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(variant_rows[0]));w.writeheader();w.writerows(variant_rows)
        result_path=out/f"{mode}_benchmark.json";log_path=out/f"{mode}_evaluation.log"
        command=[sys.executable,"evaluate.py","--csv",str(variant_manifest),"--data-root",str(out),"--split","test",
                 "--frames",str(a.frames),"--batch-size","1","--checkpoint",a.checkpoint,"--device",a.device,
                 "--variant","all","--output",str(result_path)]
        completed=subprocess.run(command,capture_output=True,text=True,check=False)
        log_path.write_text(completed.stdout+completed.stderr,encoding="utf-8",errors="replace")
        if completed.returncode:raise RuntimeError(f"Evaluation failed for {mode}; see {log_path}")
        measured=json.loads(result_path.read_text(encoding="utf-8"));comparisons[mode]={}
        for name,metrics in measured.items():
            if name not in reference:continue
            baseline=reference[name]
            comparisons[mode][name]={"accuracy":metrics["accuracy"],"fake_recall":metrics["fake_recall"],
                "macro_f1":metrics["macro_f1"],"coverage":metrics["coverage"],
                "delta_fake_recall":round(metrics["fake_recall"]-baseline["fake_recall"],4),
                "delta_macro_f1":round(metrics["macro_f1"]-baseline["macro_f1"],4),
                "delta_coverage":round(metrics["coverage"]-baseline["coverage"],4),
                "denominator":metrics["denominator"]}
        transforms.extend(metadata)
    summary={"source_manifest":str(manifest.resolve()),"split":a.split,"source_video_count":len(rows),
             "model_checkpoint":str(Path(a.checkpoint).resolve()),"reference_metrics":str(Path(a.reference_json).resolve()),
             "frames":a.frames,"device":a.device,"transformations":transforms,"comparisons":comparisons}
    (out/"transform_stability_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"summary":str((out/"transform_stability_summary.json").resolve()),"comparisons":comparisons},indent=2))


if __name__=="__main__":main()
