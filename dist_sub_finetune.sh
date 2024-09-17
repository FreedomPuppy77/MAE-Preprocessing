export IMAGENET_DIR=/data/lyh/RAF-DB/dataset
export JOB_DIR=/home/sherry/lyh/mae/RAF-DB-new/finetune
export PRETRAIN_CHKPT=/home/sherry/lyh/mae/RAF-DB-new/checkpoint-49.pth
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --batch_size 8 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 10 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}