import os
import random
from collections import OrderedDict
from torch.nn import functional as F

import torch
from torch import nn as nn
import clip
from torch.utils.data import Dataset

from dassl.config import get_cfg_default
from dassl.modeling.backbone import resnet50_clip

exist = lambda target_path: os.path.exists(target_path)


class BaseStyleGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        # a {} style of a {cls}
        self.classnames = classnames
        self.device = device
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.clip_model = clip_model
        self.enable_diverse = cfg.STYLE_GENERATOR.ENABLE_DIVERSE
        # 构建基础特征向量
        # 给每一个类别构建一个基础风格向量
        base_text_list = ["a random style of a " + s for s in classnames]

        # position_offset = [0 if len(j.split("_")) == 1 else 2 for j in self.classnames]
        self.style_position = [1 for _ in self.classnames]
        # 基础的风格向量
        self.tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        self.base_embedding = clip_model.token_embedding(self.tokenized_base_text).to("cpu")  # 将基础风格的token转为embedding
        self.style_embedding = []  # 保存k个风格
        self.stylized_base_text_encoder_out = []  # 保存只包含k个风格的文本（没有类别）

    def style_generator(self, embedding_dim=512):
        raise NotImplementedError("You must implement this function!")

    def get_stylized_embedding(self, single_base_embedding, style_position, style_id):
        assert style_id < len(self.style_embedding), "Style id is outside the length of the style list!"
        new_style_embedding = single_base_embedding.clone()
        new_style_embedding[0, style_position:style_position + 1, :] = self.style_embedding[style_id].clone()
        return new_style_embedding

    def _init_stylized_text(self, base_text=None, style_position=0):
        if base_text is None:
            base_text = "X-like style"
            style_position = 1
        base_text_list = [base_text] * self.n_style
        tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        stylized_base_text_embedding = self.clip_model.token_embedding(tokenized_base_text)  # 将基础风格的token转为embedding
        stylized_base_text_embedding[:, style_position:style_position + 1, :] = self.style_embedding
        self.stylized_base_text_encoder_out = self.clip_model.forward_text(stylized_base_text_embedding,
                                                                           tokenized_base_text).to("cpu")

    def reinit_style(self, embedding_dim=512):
        self.generate_style_embedding(embedding_dim)
        self._init_stylized_text()

    @torch.no_grad()
    def generate_style_embedding(self, embedding_dim=512):
        self.style_embedding = torch.cat([self.style_generator(embedding_dim) for _ in range(self.n_style)]).unsqueeze(
            1).to("cpu")

    def train_data(self):
        '''
        text_int:x
        text_init_tokenized:t_x
        "style_prompt":n_style种随机风格
        '''
        train_data = {"classnames": self.classnames,
                      "base_embedding": self.base_embedding.to("cpu"),
                      "tokenized_base_text": self.tokenized_base_text.to("cpu"),
                      "style_generator": self,
                      "n_cls": len(self.classnames),
                      "n_style": self.cfg.TRAINER.NUM_STYLES,
                      "style_position": self.style_position,
                      }
        return train_data


class RandomStyleGenerator(BaseStyleGenerator):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)
        # self.init_func = [nn.init.normal_, nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_normal_,
        #                   nn.init.kaiming_uniform_]

    def style_generator(self, embedding_dim=512):
        new_style = torch.empty(1, embedding_dim, dtype=torch.float)
        # init_func_id = random.randint(0, len(self.init_func) - 1)
        # self.init_func[init_func_id](new_style)
        # nn.init.normal_(new_style, std=0.02)
        new_style = nn.init.normal_(new_style, std=0.02)

        return new_style.to("cpu")


class MixStyleGenerator(BaseStyleGenerator):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)

        style_txt_path = cfg.STYLE_GENERATOR.STYLE_TXT_PATH
        assert exist(style_txt_path), f"{style_txt_path} is not exist!"
        with open(style_txt_path, "r") as style_file:
            self.base_style_list = style_file.read().splitlines()

        tokenized_base_style = torch.cat([clip.tokenize(s) for s in self.base_style_list]).to(device)

        self.base_style_embedding = clip_model.token_embedding(tokenized_base_style)[:, 1:2, :].squeeze()

    def style_generator(self, embedding_dim=512):
        _lambda = torch.distributions.Beta(0.1, 0.1).sample((self.base_style_embedding.shape[0],)).to(self.device)
        normalized_lambda = _lambda / _lambda.sum()
        normalized_lambda = normalized_lambda.view(self.base_style_embedding.shape[0], 1)
        new_style = normalized_lambda * self.base_style_embedding
        new_style = torch.sum(new_style, dim=0)
        new_style = new_style.view(1, new_style.shape[0])

        return new_style


class PromptStylerGenerator(BaseStyleGenerator):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)
        self.style_embedding = None  # to be optimized n_style*1*512
        self.weight_save_path = os.path.join(cfg.TRAINER.PROMPTSTYLER.WEIGHT_DIR_PATH,
                                             cfg.TRAINER.PROMPTSTYLER.CHECK_POINT_NAME)
        self.load_weight()

    def reinit_style(self, embedding_dim=512):
        print("PromptStylerGenerator not need to call reinit_style")

    def load_weight(self, device="cpu"):
        assert exist(self.weight_save_path), "prompt style weight path not exist!"
        # load weight
        state_dict = torch.load(self.weight_save_path, map_location=device)
        self.style_embedding = state_dict["style_embedding"]
        self.style_embedding.requires_grad_(False)
        print(f"load weight from {self.weight_save_path}")

    def style_generator(self, embedding_dim=512):
        raise "This generator is not need to call style_generator!"


class RandomMixStyleGenerator(MixStyleGenerator):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)
        self.init_func = [nn.init.normal_, nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_normal_,
                          nn.init.kaiming_uniform_]

    def style_generator(self, embedding_dim=512):
        random_choice = random.randint(0, 1)
        if random_choice == 0:
            return super().style_generator(embedding_dim)
        else:
            new_style = torch.empty(1, embedding_dim, dtype=torch.float)
            init_func_id = random.randint(0, len(self.init_func) - 1)
            self.init_func[init_func_id](new_style)
            # nn.init.normal_(new_style, std=0.02)
            return new_style.to(self.device)

class MixStyleDataset(object):
    def __init__(self, style_txt_path, clip_model, device, style_nums=80):
        assert exist(style_txt_path), f"{style_txt_path} is not exist!"
        with open(style_txt_path, "r") as style_file:
            self.base_style_list = style_file.read().splitlines()

        tokenized_base_style = torch.cat([clip.tokenize(s) for s in self.base_style_list]).to(device)

        self.base_style_embedding = clip_model.token_embedding(tokenized_base_style)[:, 1:2, :].squeeze()
        self.style_nums = style_nums
        self.device = device

    def get_style_list(self):
        return torch.cat([self.style_generator() for _ in range(self.style_nums)])

    def style_generator(self):
        _lambda = torch.distributions.Beta(0.1, 0.1).sample((self.base_style_embedding.shape[0],)).to(self.device)
        normalized_lambda = _lambda / _lambda.sum()
        normalized_lambda = normalized_lambda.view(self.base_style_embedding.shape[0], 1)
        new_style = normalized_lambda * self.base_style_embedding
        new_style = torch.sum(new_style, dim=0)
        new_style = new_style.view(1, new_style.shape[0])
        return new_style
