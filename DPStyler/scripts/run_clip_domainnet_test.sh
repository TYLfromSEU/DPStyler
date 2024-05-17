#!/bin/bash
# pip install yacs
# pip install gdown

cd ..
DATASET_ROOT=./datasets
DATASET=domainnet
NET=resnet50_clip
DATASET_YAML=domainnet_source_free
#NET_INIT=pretrain/resnet50.pth

ND=0
BATCH=128
I2_EPOCHS=100
DEVICE=0

D1=clipart
D2=infograph
D3=painting
D4=quickdraw
D5=real
D6=sketch
DATASET_NAME='DomainNet_SF'


for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer WOPA_ensemble \
    --eval-only \
    --source-domains none\
    --target-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/${DATASET}/NS80_lr8_arcface_5_0.5_E${I2_EPOCHS}_${NET}_RM_MT/seed${SEED} \
    --arcface_s 5 \
    --arcface_m 0.5 \
    --num_styles 80 \
    --txts_path dassl/txts \
    --eval_epoch 10 \
    --refresh  RandomMix \
    TRAINER.TEMPLATE_TXT True \
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.008 \
    DATASET.ROOT ${DATASET_ROOT}\
    MODEL.HEAD.NAME se_attn_sr \
   
done
