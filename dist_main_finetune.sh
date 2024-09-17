export IMAGENET_DIR=/data/lyh/RAF-DB/dataset
python main_finetune.py \
    --eval \
    --resume mae_finetuned_vit_base.pth \
    --model vit_base_patch16 \
    --batch_size 8 \
    --data_path ${IMAGENET_DIR}