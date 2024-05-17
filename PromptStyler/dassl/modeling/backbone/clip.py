import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import clip
from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class Clip(Backbone):
    def __init__(self, clip_enocder_name, device):
        super().__init__()
        self.model, self.preprocess = clip.load(clip_enocder_name, device=device)

        self.model.float()
        self.device = device
        # freeze everything
        for name, val in self.model.named_parameters():
            val.requires_grad = False
        # image part
        self._out_features = self.model.visual.output_dim
        # text part
        self.transformer = self.model.transformer
        self.positional_embedding = self.model.positional_embedding
        self.ln_final = self.model.ln_final
        self.text_projection = self.model.text_projection
        self.dtype = self.model.dtype
        self.token_embedding = self.model.token_embedding

    def forward_text_ori(self, tokenized_prompts):
        return self.model.encode_text(tokenized_prompts)

    def forward_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x).type(self.dtype)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def forward_image(self, x):
        image_features = self.model.encode_image(x)
        return image_features


@BACKBONE_REGISTRY.register()
def resnet50_clip(device, **kwargs):
    model = Clip('RN50', device)
    return model


@BACKBONE_REGISTRY.register()
def vitb16_clip(device, **kwargs):
    model = Clip('ViT-B/16', device)
    return model


@BACKBONE_REGISTRY.register()
def vitl14_clip(device, **kwargs):
    model = Clip('ViT-L/14', device)
    return model
