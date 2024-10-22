export CUDA_VISIBLE_DEVICES=0,1
export IMAGENET_DIR=/data/lyh/Affwild2/cropped_aligned
export JOB_DIR=/data/lyh/Affwild2/finetune
export PRETRAIN_CHKPT=/data/lyh/AffectNet/AffectNet_log/checkpoint-799.pth
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --batch_size 8 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 200 \
    --blr 1e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --resume /data/lyh/Affwild2/finetune/checkpoint-13.pth