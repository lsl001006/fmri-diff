CUDA_VISIBLE_DEVICES=5 python code/stageB_ldm_finetune.py \
--batch_size 32 \
--bold5000_path /data/zbh/datasets/fmri/data/BOLD5000 --dataset BOLD5000 \
--pretrain_mbm_path /data/zbh/ckpt/nips2023/frmi_pretrains/BOLD5000/fmri_encoder.pth \
--pretrain_gm_path /data/zbh/ckpt/nips2023/frmi_pretrains/ldm/label2img \
--checkpoint_path /data/zbh/code/nips2023/fmri-diffusion/fmri-diffusion/results/generation/13-04-2023-19-21-37/checkpoint_best.pth \
