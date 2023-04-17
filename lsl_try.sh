# lsl test
CUDA_VISIBLE_DEVICES=1 python code/stageB_uni_finetune_control.py \
--batch_size 8 \
--bold5000_path ./data/BOLD5000 --dataset BOLD5000 \
--checkpoint_path ../out_ckpts/BOLD5000 \
--pretrain_mbm_path ../pretrains/BOLD5000/fmri_encoder.pth \
--player_name lsl

