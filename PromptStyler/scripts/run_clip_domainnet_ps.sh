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
I2_EPOCHS=50
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
    OUT_DIR=output/${DATASET}/${NET}/PS_re_train_style/seed${SEED}

    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer PromptStylerTrainer \
    --source-domains none\
    --target-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir ${OUT_DIR} \
    --arcface_s 5\
    --arcface_m 0.5\
    --num_styles 80\
    --txts_path dassl/txts\
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.002 \
    DATASET.ROOT ${DATASET_ROOT} \
    TRAINER.PROMPTSTYLER.WEIGHT_DIR_PATH \
    "${OUT_DIR}"/checkpoint \
    TRAINER.PROMPTSTYLER.TRAIN_STYLE  True


    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer WOPA_clip \
    --source-domains none\
    --target-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir output/${DATASET}/${NET}/PS_re_train_cls/seed${SEED} \
    --arcface_s 5\
    --arcface_m 0.5\
    --num_styles 80\
    --txts_path dassl/txts\
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.005 \
    DATASET.ROOT ${DATASET_ROOT} \
    STYLE_GENERATOR.NAME "PromptStylerGenerator" \
    TRAINER.PROMPTSTYLER.WEIGHT_DIR_PATH \
    "${OUT_DIR}"/checkpoint
    
done
