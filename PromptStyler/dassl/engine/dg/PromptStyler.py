import os

import torch
import torch.nn as nn
import clip
from torch.nn import functional as F
from dassl.modeling.ops import InfoNCE

exist = lambda target_path: os.path.exists(target_path)


class PromptStyler(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.classnames = classnames
        self.device = device
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model = clip_model
        self.weight_save_path = os.path.join(cfg.TRAINER.PROMPTSTYLER.WEIGHT_DIR_PATH,
                                             cfg.TRAINER.PROMPTSTYLER.CHECK_POINT_NAME)
        # raw template: a X style of a
        self.prompt_style_template = "a X style of a"
        self.tokenized_prompt_style_template = clip.tokenize(self.prompt_style_template).to(self.device)  # t_x  1*77
        with torch.no_grad():
            self.prompt_style_template_embedding = self.clip_model.token_embedding(
                self.tokenized_prompt_style_template)  # x  1*77*512
        # content template:"class"
        # self.prompt_content_template = [f"a photo of a {cls}" for cls in classnames]
        self.prompt_content_template = classnames

        self.tokenized_prompt_content_template = clip.tokenize(self.prompt_content_template).to(device)  # t_x  7*77
        with torch.no_grad():
            self.prompt_content_output = self.clip_model.forward_text_ori(
                self.tokenized_prompt_content_template)  # output 7*1024
            self.prompt_content_output_norm = F.normalize(self.prompt_content_output, p=2, dim=1)  # normalize

        # raw template: a X style of a CLS
        self.init_template(self.classnames, "a X style of a CLS")
        self.tokenized_prompt_s_c = self.tokenized_template_base_text[0].to(self.device)  # t_x 7*77
        self.prompt_s_c_embedding = self.template_base_text[0].to(self.device)  # x 7*77*512
        self.train_flag = cfg.TRAINER.PROMPTSTYLER.TRAIN_STYLE
        if not self.train_flag:
            self.style_embedding = nn.Parameter(
                torch.empty(self.n_style, 1, self.ctx_dim, dtype=torch.float))  # to be optimized n_style*1*512
            self.load_weight()
            self.style_embedding.requires_grad_(False)
            print(self.style_embedding.device)
        else:
            ctx_vectors = torch.empty(self.n_style, 1, self.ctx_dim, dtype=torch.float).to(self.device)
            ctx_vectors.requires_grad_(True)
            ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
            self.style_embedding = nn.Parameter(ctx_vectors)  # to be optimized n_style*1*512
            self.style_content_loss = InfoNCE()

    def save_style_embedding(self, weight_path):
        state_dict = self.state_dict()
        new_state_dict = {"style_embedding": state_dict["style_embedding"]}
        torch.save(new_state_dict, weight_path)
        print("save style embedding weight")

    def load_weight(self, device="cpu"):
        assert exist(self.weight_save_path), "prompt style weight path not exist!"
        state_dict = torch.load(self.weight_save_path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

        print(f"load weight from {self.weight_save_path}")

    def init_template(self, classnames, template_txt):
        self.template_base_text_list = []
        self.style_position = []
        self.tokenized_template_base_text = []
        self.template_base_text = []
        self.template_nums = 1
        template_txt_split = template_txt.split()
        style_position = template_txt_split.index("X")
        cls_position = template_txt_split.index("CLS")
        style_after_cls_flag = style_position > cls_position
        style_position_offset = [1 if "_" not in cls and not style_after_cls_flag else 3 for cls in classnames]

        template_base_text = [template_txt.replace('CLS', s) for s in classnames]
        tokenized_template_base_text = torch.cat([clip.tokenize(p) for p in template_base_text]).to(self.device)
        with torch.no_grad():
            template_base_text = self.clip_model.token_embedding(
                tokenized_template_base_text)  #
        self.tokenized_template_base_text.append(tokenized_template_base_text.to('cpu'))  # t_x
        self.template_base_text.append(template_base_text.to('cpu'))  # x

        self.template_base_text_list += template_base_text

        style_position = template_txt_split.index("X")
        self.style_position.append([style_position + i for i in style_position_offset])

    def get_stylized_embedding(self, template_idx, class_idx, style_idx):
        assert style_idx < len(self.style_embedding), "Style id is outside the length of the style list!"
        assert class_idx < len(self.classnames), "Class id is outside the length of the class list!"
        assert template_idx < self.template_nums, "Template id is outside the length of the template list!"
        token_template_base = self.tokenized_template_base_text[template_idx]
        template_base = self.template_base_text[template_idx]
        token_base = token_template_base[class_idx:class_idx + 1, :].clone()
        base = template_base[class_idx:class_idx + 1, :, :].clone()
        style_init_embedding = self.style_embedding[style_idx:style_idx + 1, :, :]
        style_position = self.style_position[template_idx][class_idx]
        base[:, style_position:style_position + 1, :] = style_init_embedding

        return base, token_base

    def forward(self, style_idx):
        # *****init*****
        current_style = self.style_embedding[style_idx:style_idx + 1, :, :]  # 1*1*512
        if style_idx > 0:
            # **************style diversity loss**************
            prefix = self.prompt_style_template_embedding[:, :2, :]  # 1*2*512
            suffix = self.prompt_style_template_embedding[:, 3:, :]  # 1*74*512
            current_prompt = torch.cat(
                [
                    prefix,  # (1, 2, dim)
                    current_style,  # (1, 1, dim)
                    suffix,  # (1, *, dim)
                ],
                dim=1,
            )  # 1*77*512
            current_prompt_output = self.clip_model.forward_text(current_prompt,
                                                                 self.tokenized_prompt_style_template)  # 1*1024
            with torch.no_grad():
                before_styles = self.style_embedding[:style_idx, :, :]  # style_idx*1*512,
                before_prefix = prefix.repeat(style_idx, 1, 1)  # style_idx*2*512
                before_suffix = suffix.repeat(style_idx, 1, 1)  # style_idx*74*512
                before_prompts = torch.cat(
                    [
                        before_prefix,  # (style_idx, 2, dim)
                        before_styles,  # (style_idx, 1, dim)
                        before_suffix,  # (style_idx, *, dim)
                    ],
                    dim=1,
                )  # style_idx*77*512
                before_tokenized_prompts = self.tokenized_prompt_style_template.repeat(style_idx, 1)  # style_idx*77
                before_prompts_output = self.clip_model.forward_text(before_prompts,
                                                                     before_tokenized_prompts)  # style_idx*1024
            # normalize
            current_prompt_output_norm = F.normalize(current_prompt_output, p=2, dim=1)
            before_prompts_output_norm = F.normalize(before_prompts_output, p=2, dim=1)
            current_prompt_output_norm = current_prompt_output_norm.repeat(before_prompts_output_norm.size(0), 1)

            cos_sim = F.cosine_similarity(current_prompt_output_norm, before_prompts_output_norm, dim=1)

            style_diversity_loss = torch.abs(cos_sim)
            style_diversity_loss = style_diversity_loss.mean()
        else:
            style_diversity_loss = torch.tensor(0.0)
        # *********Content consistency loss**************
        sc_prompts_list = []
        for cls_idx in range(self.n_cls):
            style_position_idx = self.style_position[0][cls_idx]
            sc_prefix = self.prompt_s_c_embedding[cls_idx:cls_idx + 1, :style_position_idx, :]
            sc_suffix = self.prompt_s_c_embedding[cls_idx:cls_idx + 1, style_position_idx + 1:, :]  # 7*74*512
            sc_prompts_list.append(torch.cat(
                [
                    sc_prefix,  # (1, 2, dim)
                    current_style,  # (1, 1, dim)
                    sc_suffix,  # (1, *, dim)
                ],
                dim=1,
            ))
        sc_prompts = torch.cat(sc_prompts_list, dim=0)  # 7*77*512
        sc_prompts_output = self.clip_model.forward_text(sc_prompts,
                                                         self.tokenized_prompt_s_c)  # 7*1024
        sc_prompts_output_norm = F.normalize(sc_prompts_output, p=2, dim=1)
        zimm = F.cosine_similarity(sc_prompts_output_norm.unsqueeze(1), self.prompt_content_output_norm.unsqueeze(0),
                                   dim=2)
        exp_zimm = torch.exp(zimm)
        per_zimm = exp_zimm / exp_zimm.sum(dim=1, keepdim=True)

        content_consistency_loss_dj = -torch.log(per_zimm.diag()).mean()

        return style_diversity_loss, content_consistency_loss_dj
