export IMAGENET_DIR=/data/lyh/Affwild2_examples/aligned
export JOB_DIR=/home/sherry/lyh/mae/Affwild2_examples
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --use_volta32 \
    --batch_size 8 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} 