#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
YOLOX_ROOT = ROOT / "external" / "YOLOX"
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from exps.default.yolox_nano import Exp as NanoExp  # type: ignore


class Exp(NanoExp):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (20, 20)
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 10
        self.print_interval = 20
        self.basic_lr_per_img = 0.01 / 64.0
        self.enable_mixup = False
        self.mosaic_prob = 1.0
        self.no_aug_epochs = 10
        self.data_dir = str(ROOT / "datasets" / "processed" / "deeppcb_coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
