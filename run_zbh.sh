CUDA_VISIBLE_DEVICES=4 python code/stageB_uni_finetune_control.py \
--batch_size 8 \
--bold5000_path /data/zbh/datasets/fmri/data/BOLD5000 --dataset BOLD5000 \
--checkpoint_path /data/zbh/output/nips2023/fmri_diff/BOLD5000 \
--pretrain_mbm_path /data/zbh/ckpt/nips2023/frmi_pretrains/BOLD5000/fmri_encoder.pth \
--player_name zbh

# CUDA_VISIBLE_DEVICES=4 python code/inference_test.py \
# --batch_size 8 \
# --bold5000_path /expand_data/datasets/fmri/data/BOLD5000 --dataset BOLD5000 \
# --checkpoint_path /data/zbh/output/output/nips2023/fmri_diff/BOLD5000 \
# --pretrain_mbm_path /expand_data/zbh/ckpt/frmi_pretrains/BOLD5000/fmri_encoder.pth \
# --pretrain_gm_path /expand_data/zbh/ckpt/frmi_pretrains/ldm/label2img \
# --player_name zbh
