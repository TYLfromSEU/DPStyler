import torch
import torch.nn as nn
import clip
import os
import random

exist = lambda target_path: os.path.exists(target_path)

class PromptGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device):
        super().__init__()
        
        self.classnames=classnames
        self.device=device
        self.cfg=cfg
        self.n_cls = len(classnames)
        self.n_style = cfg.TRAINER.NUM_STYLES
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.clip_model=clip_model
        #for random init style
        self.init_func = [nn.init.normal_, nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_normal_,
                          nn.init.kaiming_uniform_]
        #for mix init style
        style_txt_path = cfg.STYLE_GENERATOR.STYLE_TXT_PATH
        assert exist(style_txt_path), f"{style_txt_path} is not exist!"
        with open(style_txt_path, "r") as style_file:
            self.base_style_list = style_file.read().splitlines()
        tokenized_base_style = torch.cat([clip.tokenize(s) for s in self.base_style_list]).to(device)
        with torch.no_grad():
            self.base_style_embedding = clip_model.token_embedding(tokenized_base_style)[:, 1:2, :].squeeze().to("cpu")
        # Initializing the template
        self.init_template(self.cfg,self.classnames)
        #Initialization style
        self.init_style()
        #Initialize the domain loss classification weight
        self.init_stylized_text()



    def init_template(self,cfg,classnames):
        #Constructing templates
        self.template_base_text_list = []
        self.style_position = []
        self.tokenized_template_base_text=[]
        self.template_base_text=[]
        position_offset=[ 0 if len(j.split("_"))==1 else 2 for j in self.classnames]
        if cfg.TRAINER.TEMPLATE_TXT:  
            template_path = cfg.STYLE_GENERATOR.TEMPLATE_PATH
            assert exist(template_path), f"{template_path} is not exist"
            with open(template_path, "r", encoding="utf-8") as f:
                template_list = f.read().splitlines()
            self.template_nums = len(template_list)
        else:
            template_list=["a CLS in a X style"]
            #template_list=["a photo of a CLS with X-like style"]
            self.template_nums = 1
        for template_txt in template_list:
            template_txt_split = template_txt.split()
            template_base_text = [template_txt.replace('CLS', s) for s in classnames]# create num cls sentences
            #Convert sentences with the same template into tokens and embeddings.
            tokenized_template_base_text = torch.cat([clip.tokenize(p) for p in template_base_text]).to(self.device)
            with torch.no_grad():
                template_base_text = self.clip_model.token_embedding(tokenized_template_base_text)  # Convert tokens in the style of x to embeddings.
            self.tokenized_template_base_text.append(tokenized_template_base_text.to('cpu')) #t_x
            self.template_base_text.append(template_base_text.to('cpu')) #x
            #Store the position of the style.
            self.template_base_text_list += template_base_text
            style_position = template_txt_split.index("X") + 1  # 1 as the offset position.

            if template_txt_split.index("X")>template_txt_split.index("CLS"):
                self.style_position.append([style_position+i for i in position_offset])
            else:
                self.style_position.append([style_position for _ in range(len(classnames))])


    def init_style(self):
        self.style_embedding=torch.empty(self.n_style, 1, self.ctx_dim, dtype=self.dtype) # Create n_style random styles.
        nn.init.normal_(self.style_embedding, std=0.02)

    def refresh_style(self):
        new_styles=[]
        for _ in range(self.n_style):
            if self.cfg.TRAINER.REFRESH=="RandomMix":
                random_choice = random.randint(0, 1)
                if random_choice == 0:  # Mix style
                    new_style=self.mix_generator()
                else: # random
                    new_style = torch.empty(1, self.ctx_dim, dtype=torch.float)
                    init_func_id = random.randint(0, len(self.init_func) - 1)
                    self.init_func[init_func_id](new_style)
                
            elif self.cfg.TRAINER.REFRESH=="Random":
                new_style = torch.empty(1, self.ctx_dim, dtype=torch.float)
                init_func_id = random.randint(0, len(self.init_func) - 1)
                self.init_func[init_func_id](new_style)
        
            elif self.cfg.TRAINER.REFRESH=="Mix":
                new_style=self.mix_generator()
            new_style=new_style.to('cpu')
            new_styles.append(new_style)
        self.style_embedding = torch.stack(new_styles)
        self.init_stylized_text()

    def mix_generator(self):
        _lambda = torch.distributions.Beta(0.1, 0.1).sample((self.base_style_embedding.shape[0],))
        normalized_lambda = _lambda / _lambda.sum()  # Normalize the random numbers.
        normalized_lambda = normalized_lambda.view(self.base_style_embedding.shape[0], 1)
        new_style = normalized_lambda * self.base_style_embedding
        new_style = torch.sum(new_style, dim=0)
        new_style = new_style.view(1, new_style.shape[0])
        return new_style

    def init_stylized_text(self, base_text=None, style_position=0):# Used for domain loss.
        if base_text is None:
            base_text = "x-like style"
            style_position = 1
        base_text_list = [base_text] * self.n_style
        tokenized_base_text = torch.cat([clip.tokenize(p) for p in base_text_list]).to(self.device)
        with torch.no_grad():
            stylized_base_text_embedding = self.clip_model.token_embedding(tokenized_base_text)  # Convert basic-style tokens into embeddings.
        stylized_base_text_embedding[:, style_position:style_position + 1, :] = self.style_embedding
        self.stylized_base_text_encoder_out = self.clip_model.forward_text(stylized_base_text_embedding,
                                                                           tokenized_base_text)
    
    def get_stylized_embedding(self, template_idx,class_idx,style_idx):
        assert style_idx < len(self.style_embedding), "Style id is outside the length of the style list!"
        assert class_idx < len(self.classnames), "Class id is outside the length of the class list!"
        assert template_idx < self.template_nums, "Template id is outside the length of the template list!"
        # Select the template at index template_idx.
        token_template_base=self.tokenized_template_base_text[template_idx]
        template_base=self.template_base_text[template_idx]
        # Select the class at index class_idx.
        token_base=token_template_base[class_idx:class_idx+1,:].clone()
        base=template_base[class_idx:class_idx+1,:, :].clone()
        # Select the style at index style_idx.
        style_init_embedding=self.style_embedding[style_idx:style_idx+1, :, :]
        # Select the style position at index class_idx.
        style_position=self.style_position[template_idx][class_idx]
        # Replace the style.
        base[:,style_position:style_position+1,:]=style_init_embedding

        return base ,token_base

        