#!/bin/bash
# pip install yacs
# pip install gdown

cd ..


DATASET_ROOT=./datasets
DATASET=pacs
NET=resnet50_clip
DATASET_YAML=pacs_source_free
DEVICE=0

ND=0
BATCH=128
I2_EPOCHS=100


if [ ${DATASET} == 'pacs' ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch

    DATASET_NAME='PACS_SF'
    
elif [ ${DATASET} == 'office_home_dg' ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
    
    DATASET_NAME='OfficeHomeDG'
    
fi

for SEED in 1 2 3
do
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer WOPA_clip \
    --source-domains none\
    --target-domains ${D1} ${D2} ${D3} ${D4} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/${DATASET}/NS80_lr8_arcface_5_0.5_E${I2_EPOCHS}_${NET}_M_wohead/seed${SEED} \
    --arcface_s 5 \
    --arcface_m 0.5 \
    --num_styles 80 \
    --txts_path dassl/txts \
    --eval_epoch 100 \
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
