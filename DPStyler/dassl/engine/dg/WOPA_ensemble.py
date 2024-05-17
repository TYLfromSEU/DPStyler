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
from dassl.data import DataManager,DataManager_sf
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import AngularPenaltySMLoss,EntropyMaximization,InfoNCE
from dassl.modeling.head import se_attn_sr
from dassl.evaluation import build_evaluator
from dassl.engine.dg.PromptGenerator import PromptGenerator



class clip_net_arcface(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface',**kwargs):
        super(clip_net_arcface, self).__init__()
        self.device=device
        # create clip backbone
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            device=self.device,
            **kwargs
        )
        self.fdim=self.backbone._out_features
        # create head
        self.head = None
        if model_cfg.HEAD.NAME=='se_attn_sr':
            print("******init head******")
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                num_channels=self.fdim,
                reduction_ratio=model_cfg.HEAD.REDUCTION_RATIO,
                **kwargs
            )
            self.fdim = self.head.out_features
        else:
            print("******without head******")
        # create classifier
        self.adms_loss = AngularPenaltySMLoss(self.fdim, num_classes, loss_type=loss_type,s=cfg.ARCFACE_S, m=cfg.ARCFACE_M)

    def forward_text(self, x,t_x,style_embeddings,labels,norm=False):
        #backbone
        backbone_output = self.backbone.forward_text(x,t_x)
        embed_output=backbone_output/backbone_output.norm(dim=-1,keepdim=True)
        #head
        if self.head is not None:
            embed_output=self.head(embed_output) 
        if norm:
            embed_output=embed_output/embed_output.norm(dim=-1,keepdim=True)
        #classification
        y_class=self.adms_loss.fc(embed_output)
        y_loss=self.adms_loss(embed_output,labels)
        y_domain=self.predictor(embed_output,style_embeddings)
        return y_class,y_loss,y_domain

    def forward_img_1(self,x,norm=False):
        # #backbone
        # backbone_output=self.backbone.forward_image(x)
        embed_output=x/x.norm(dim=-1,keepdim=True)
        #head
        if self.head is not None:
            embed_output=self.head(embed_output) 
        if norm:
            embed_output=embed_output/embed_output.norm(dim=-1,keepdim=True)
        #classification
        y_class = self.adms_loss.fc(embed_output)
        return y_class
    
    def forward_img(self,x,norm=False):
        #backbone
        backbone_output=self.backbone.forward_image(x)
        embed_output=backbone_output/backbone_output.norm(dim=-1,keepdim=True)
        #head
        if self.head is not None:
            embed_output=self.head(embed_output) 
        if norm:
            embed_output=embed_output/embed_output.norm(dim=-1,keepdim=True)
        #classification
        y_class = self.adms_loss.fc(embed_output)
        return y_class
    
    def predictor(self,feat,teat):
        feat_p = feat/feat.norm(dim=-1,keepdim=True)
        teat_p = teat/teat.norm(dim=-1, keepdim=True)
        scores =  (100.0 * torch.matmul(feat_p,teat_p.detach().T))
        scores = torch.cat([scores,torch.zeros(scores.shape[0],1,device=scores.device)],1) 
        return scores
    
class head(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, device, loss_type='arcface',**kwargs):
        super(head, self).__init__()
        if model_cfg.HEAD.NAME=='se_attn_sr':
            print("******init head******")
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                num_channels=1024,
                reduction_ratio=model_cfg.HEAD.REDUCTION_RATIO,
                **kwargs
            )
            self.fdim = self.head.out_features
        else:
            print("******without head******")
        # create classifier
        self.adms_loss = AngularPenaltySMLoss(self.fdim, num_classes, loss_type=loss_type,s=cfg.ARCFACE_S, m=cfg.ARCFACE_M)

    def forward_img(self,x,norm=False):
        #backbone
        #backbone_output=self.backbone.forward_image(x)
        embed_output=x/x.norm(dim=-1,keepdim=True)
        #head
        if self.head is not None:
            embed_output=self.head(embed_output) 
        if norm:
            embed_output=embed_output/embed_output.norm(dim=-1,keepdim=True)
        #classification
        y_class = self.adms_loss.fc(embed_output)
        return y_class


@TRAINER_REGISTRY.register()
class WOPA_ensemble(TrainerX):
    def __init__(self,cfg):
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
        self.init_train_data()
        self.build_data_loader()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

        self.enm_loss = EntropyMaximization()
        self.infonce_loss=InfoNCE(negative_mode='paired')

    def build_model(self):
        cfg = self.cfg
        print("Building model")

        if self.cfg.DATASET.NAME=='PACS_SF':
            self.num_classes=7
        elif self.cfg.DATASET.NAME=='OfficeHomeDG_SF':
            self.num_classes=65
        elif self.cfg.DATASET.NAME=='VLCS_SF':
            self.num_classes=5
        elif self.cfg.DATASET.NAME=='DomainNet_SF':
            self.num_classes=345

        self.model=[]
        self.model_1 = clip_net_arcface(cfg, cfg.MODEL, self.num_classes,self.device)
        self.model_2=head(cfg, cfg.MODEL, self.num_classes,self.device)
        self.model_3=head(cfg, cfg.MODEL, self.num_classes,self.device)

        # TODO: Please modify the model weight path.
        load_pretrained_weights(self.model_1, 'model_path_template_1.pth.tar')
        load_pretrained_weights(self.model_2, 'model_path_template_2.pth.tar')
        load_pretrained_weights(self.model_3, 'model_path_template_3.pth.tar')
        
        
        self.model_1.to(self.device)
        self.model_1.eval()
        self.model_2.to(self.device)
        self.model_2.eval()
        self.model_3.to(self.device)
        self.model_3.eval()


        self.model.append(self.model_1)
        self.model.append(self.model_2)
        self.model.append(self.model_3)

        print(f"# params: {count_num_param(self.model_1.backbone.model.visual):,}")
        print(f"# params: {count_num_param(self.model_1.backbone):,}")
        print(f"# params: {count_num_param(self.model_1):,}")
        print(f"# params: {count_num_param(self.model_2):,}")
        print(f"# params: {count_num_param(self.model_3):,}")

        self.optim_1 = build_optimizer(self.model_1, cfg.OPTIM)
        self.sched_1 = build_lr_scheduler(self.optim_1, cfg.OPTIM)
        self.register_model("model", self.model_1, self.optim_1, self.sched_1)

        
    def init_train_data(self):
        txts_dir_path=self.cfg.TXTS_PATH
        txt_path=os.path.join(txts_dir_path,self.cfg.DATASET.NAME+'.txt')
        
        with open(txt_path,'r') as f:
            lines = f.read().splitlines()
        class_dict = {index: value for index, value in enumerate(lines)}
        classnames=list(class_dict.values())
        self.num_classes=len(classnames)
        self.prompt_generater=PromptGenerator(self.cfg,classnames,self.model_1.backbone,self.device)

    def build_data_loader(self):
        dm = DataManager_sf(self.cfg,self.prompt_generater)

        self.train_loader_x = dm.train_loader_x
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm

   
    def model_inference(self, input):
        output=[]
        x=self.model[0].backbone.forward_image(input)
        output.append(self.model[0].forward_img_1(x))
        for i in range(1,len(self.model)):
            output.append(self.model[i].forward_img(x))
        return output
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        
        self.set_model_mode("eval")
        self.evaluator.reset()
        result=[]
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        
        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            time_start = time.time()
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process_ensemble(output,label)
            elapsed = round(time.time() - time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Elapsed: {elapsed}")
            results = self.evaluator.evaluate()
            
            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()
        
        return result
    
    

    