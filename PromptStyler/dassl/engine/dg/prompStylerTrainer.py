import datetime
import time
import torch
import os

from collections import OrderedDict
from dassl.engine.dg.PromptStyler import PromptStyler
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
)
from dassl.modeling import build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerBase


@TRAINER_REGISTRY.register()
class PromptStylerTrainer(TrainerBase):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        self.n_style = cfg.TRAINER.NUM_STYLES
        weight_dir_path = cfg.TRAINER.PROMPTSTYLER.WEIGHT_DIR_PATH
        if not os.path.exists(weight_dir_path):
            os.makedirs(weight_dir_path)
        self.weight_save_path = os.path.join(weight_dir_path, cfg.TRAINER.PROMPTSTYLER.CHECK_POINT_NAME)
        self.init_train_data()
        self.build_model()

    def init_train_data(self):
        txts_dir_path = self.cfg.TXTS_PATH
        txt_path = os.path.join(txts_dir_path, self.cfg.DATASET.NAME + '.txt')

        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        self.classnames = list(class_dict.values())
        self.num_classes = len(self.classnames)

    def build_model(self):
        cfg = self.cfg
        print("Building model")
        self.clip_model = build_backbone(cfg.MODEL.BACKBONE.NAME,
                                         verbose=cfg.VERBOSE,
                                         device=self.device,
                                         )
        self.clip_model.to(self.device)
        self.model = PromptStyler(cfg, self.classnames, self.clip_model, self.device)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM, max_epoch=self.n_style * self.max_epoch)
        self.register_model("model", self.model, self.optim)

    def train(self):
        start_time = time.time()
        self.model.train()
        for style_idx in range(0, self.n_style):
            for epoch in range(self.max_epoch):
                self.optim.zero_grad()
                # model forward
                style_diversity_loss, content_consistency_loss = self.model(style_idx)
                # *********Total loss**************
                total_loss = style_diversity_loss + content_consistency_loss
                total_loss.backward()
                self.optim.step()
                self.sched.step()

                # print training info...
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    current_lr = self.optim.param_groups[0]["lr"]
                    info = []
                    info += [f"style_idx {style_idx}"]
                    info += [f"epoch [{epoch + 1}/{self.max_epoch}]"]
                    info += [f"total loss {total_loss.item()}"]
                    info += [f"style_diversity_loss {style_diversity_loss.item()}"]
                    info += [f"content_consistency_loss {content_consistency_loss.item()}"]
                    info += [f"lr {current_lr:.4e}"]
                    print(" ".join(info))
        # save model
        self.model.eval()
        self.model.save_style_embedding(self.weight_save_path)
        # Show elapsed time
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")
