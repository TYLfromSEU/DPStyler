import datetime
import time
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import clip
from collections import OrderedDict
from dassl.data import DataManager, DataManager_sf
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.engine.dg.style_generator import RandomStyleGenerator, BaseStyleGenerator, MixStyleGenerator, \
    RandomMixStyleGenerator, PromptStylerGenerator
from dassl.modeling import build_head, build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import AngularPenaltySMLoss, EntropyMaximization, InfoNCE
from dassl.modeling.head import se_attn_sr
from dassl.evaluation import build_evaluator


class clip_net(nn.Module):
    def __init__(self, cfg, model_cfg, device, **kwargs):
        super().__init__()

        self.device = device
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
            **kwargs
        )
        self.fdim = self.backbone._out_features
        self.head = None
        if model_cfg.HEAD.NAME == 'se_attn_sr':
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                num_channels=self.fdim,
                reduction_ratio=model_cfg.HEAD.REDUCTION_RATIO,
                **kwargs,
            )
            self.fdim = self.head.out_features

    def forward_text(self, x, t_x):
        t = self.backbone.forward_text(x, t_x)  # text embed without norm
        t = t / t.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t = self.head(t)  # text embed after head without norm
        return t

    def forward_img(self, x):  # for test
        t_img = self.backbone.forward_image(x)
        t_img = t_img / t_img.norm(dim=-1, keepdim=True)  # norm after embed
        if self.head is not None:
            t_img = self.head(t_img)  # img embed after head without norm
        return t_img


class clip_net_arcface(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface'):
        super(clip_net_arcface, self).__init__()
        self.embedlayers = clip_net(cfg, model_cfg, device)
        in_features = self.embedlayers.fdim  # embed dim
        self.adms_loss = AngularPenaltySMLoss(in_features, num_classes, loss_type=loss_type, s=cfg.ARCFACE_S,
                                              m=cfg.ARCFACE_M)

    def forward_text(self, x, t_x, stylize_base_text_embedding, labels, norm=False):
        text_encoder_output = self.embedlayers.forward_text(x, t_x)  # without normalize after head
        if norm:
            text_encoder_output = text_encoder_output / text_encoder_output.norm(dim=-1, keepdim=True)  # norm
        y_pred = self.adms_loss.fc(text_encoder_output)
        y_loss = self.adms_loss(text_encoder_output, labels)

        y_domain = self.predictor(text_encoder_output,
                                  stylize_base_text_embedding) if stylize_base_text_embedding is not None else None
        return y_pred, y_loss, y_domain, text_encoder_output

    def forward_img(self, x, norm=False):
        t_img = self.embedlayers.forward_img(x)  # without normalize after head
        if norm:
            t_img = t_img / t_img.norm(dim=-1, keepdim=True)  # norm
        y_class = self.adms_loss.fc(t_img)
        return y_class

    def predictor(self, feat, teat):
        feat_p = feat / feat.norm(dim=-1, keepdim=True)
        teat_p = teat / teat.norm(dim=-1, keepdim=True)
        scores = (100.0 * torch.matmul(feat_p, teat_p.detach().T))
        scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], 1)
        return scores


@TRAINER_REGISTRY.register()
class WOPA_clip(TrainerX):
    def __init__(self, cfg):
        # super().__init__(cfg)
        self.style_generator: BaseStyleGenerator = None
        self.num_classes = None
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg

        self.build_model()
        self.build_train_data()
        self.build_data_loader()

        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = ([0, 0, 0, 0], 0)

        self.enm_loss = EntropyMaximization()

        self.infonce_loss = InfoNCE(negative_mode='paired')

    def build_model(self):
        cfg = self.cfg
        print("Building model")

        if self.cfg.DATASET.NAME == 'PACS_SF':
            self.num_classes = 7
        elif self.cfg.DATASET.NAME == 'OfficeHomeDG_SF':
            self.num_classes = 65
        elif self.cfg.DATASET.NAME == 'VLCS_SF':
            self.num_classes = 5
        elif self.cfg.DATASET.NAME == 'DomainNet_SF':
            self.num_classes = 345
        elif self.cfg.DATASET.NAME == 'TerraIncognita_SF':
            self.num_classes = 10
        else:
            raise f"{self.cfg.DATASET.NAME} dataset is not support!"

        self.model = clip_net_arcface(cfg, cfg.MODEL, self.num_classes, self.device)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# adms_loss params: {count_num_param(self.model.adms_loss):,}")
        if self.model.embedlayers.head is not None:
            print(f"# head params: {count_num_param(self.model.embedlayers.head):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

    def forward_backward(self, batch_data):
        input_stylized_embedding, input_tokenized_base_text, target = self.parse_batch_train(batch_data)
        y_pred, y_loss, y_domain, _ = self.model.forward_text(input_stylized_embedding, input_tokenized_base_text,
                                                              None, target,
                                                              norm=True)
        loss = y_loss

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "y_loss": y_loss.item(),
            "acc": compute_accuracy(y_pred, target)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input_stylized_embedding = batch["stylized_embedding"]
        input_tokenized_base_text = batch["tokenized_base_text"]
        target = batch["label"]
        input_stylized_embedding = input_stylized_embedding.to(self.device)
        input_tokenized_base_text = input_tokenized_base_text.to(self.device)
        target = target.to(self.device)

        return input_stylized_embedding, input_tokenized_base_text, target

    def build_data_loader(self):
        train_data = self.style_generator.train_data()
        dm = DataManager_sf(self.cfg, train_data)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        result = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()
        mean_acc = np.mean(result)
        is_best = False
        if self.best_result[-1] < mean_acc:
            self.best_result = (result, mean_acc)
            is_best = True
        return result, is_best


    def after_epoch(self):

        result, is_best = self.test()
        if is_best:
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=result,
                model_name="model-best.pth.tar"
            )
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

    def train(self):
        self.before_train()
        self.style_generator.reinit_style()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            if self.epoch % 2 == 0:
                self.after_epoch()
        self.after_train()

    def model_inference(self, input):
        return self.model.forward_img(input, norm=True)

    def build_train_data(self):

        txts_dir_path = self.cfg.TXTS_PATH
        txt_path = os.path.join(txts_dir_path, self.cfg.DATASET.NAME + '.txt')

        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        classnames = list(class_dict.values())
        self.num_classes = len(classnames)
        assert self.cfg.STYLE_GENERATOR.NAME in globals()
        self.style_generator = globals()[self.cfg.STYLE_GENERATOR.NAME](self.cfg, classnames,
                                                                        self.model.embedlayers.backbone, self.device)
