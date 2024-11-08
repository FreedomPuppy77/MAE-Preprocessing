export IMAGENET_DIR=/data/lyh/AffectNet
export JOB_DIR=/data/lyh/AffectNet/AffectNet_log
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --use_volta32 \
    --batch_size 8 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --resume /data/lyh/AffectNet/AffectNet_log/checkpoint-480.pth
