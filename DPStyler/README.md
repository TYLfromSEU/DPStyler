# DPStyler: Dynamic PromptStyler for Source-Free Domain Generalization
![](./images\method.png)

## Datasets

- PACS: https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE
- VLCS: http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file
- OfficeHome:  https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
- DomainNet:  http://ai.bu.edu/DomainNet/

**Note: After downloading the data, unzip it into the 'datasets' folder.**

## Requirements

**How to Start:**
```shell
conda env create -f environment.yml
```

## Training & Inference 

Here, taking the training and inference of PACS as an example:
```shell
# Training:
cd scripts
sh run_clip_pacs.sh
```
Please note that the current default text template used for training is 'a CLS in a X style'. 
If you intend to perform multi-model ensemble inference, modify the text in 'DPStyler/dassl/txts/text_template.txt'.

Here are three text templates used in this paper:

1. a CLS in a X style
2. a X style of a CLS
3. a photo of a CLS with X like style

---

```shell
# Inference:
cd scripts
sh run_clip_pacs_test.sh
```
