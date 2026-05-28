#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 OpenKE 的官方 TransE 真实训练与链路预测评估。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_times", type=int, default=300)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    openke_root = project_root.parents[1] / "openkg" / "OpenKE"
    data_dir = project_root / "reports" / "real_comparison_data" / "openke_policykg"
    out_dir = project_root / "reports" / "real_comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(openke_root))

    from openke.config import Trainer, Tester  # type: ignore
    from openke.module.loss import MarginLoss  # type: ignore
    from openke.module.model import TransE  # type: ignore
    from openke.module.strategy import NegativeSampling  # type: ignore
    from openke.data import TrainDataLoader, TestDataLoader  # type: ignore

    use_gpu = bool(args.use_gpu and torch.cuda.is_available())

    train_dataloader = TrainDataLoader(
        in_path=str(data_dir) + "/",
        nbatches=args.nbatches,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
    )
    test_dataloader = TestDataLoader(str(data_dir) + "/", "link")

    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        p_norm=1,
        norm_flag=True,
    )
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.train_times,
        alpha=args.alpha,
        use_gpu=use_gpu,
    )
    trainer.run()

    ckpt = out_dir / "openke_transe.ckpt"
    transe.save_checkpoint(str(ckpt))
    transe.load_checkpoint(str(ckpt))

    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=use_gpu)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

    result = {
        "model": "OpenKE-TransE",
        "train_times": args.train_times,
        "dim": args.dim,
        "alpha": args.alpha,
        "use_gpu": use_gpu,
        "metrics": {
            "MRR": float(mrr),
            "MR": float(mr),
            "Hits@10": float(hit10),
            "Hits@3": float(hit3),
            "Hits@1": float(hit1),
        },
        "checkpoint": str(ckpt),
        "data_dir": str(data_dir),
    }
    out_json = out_dir / "openke_transe_results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

