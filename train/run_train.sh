#!/bin/bash

export PYTHONPATH="./src:./:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"



export CUDA_VISIBLE_DEVICES="1" && python -m accelerate.commands.launch --config_file=./pipeline/accelerate_configs/accelerate_config_zero2.yaml --main_process_port=29901  pipeline/train/instruction_following.py --pretrained_model_name_or_path=luodian/OTTER-MPT1B-RPJama-Init --mimicit_path=mimic-it/LA/LADD_instructions.json --images_path=mimic-it/LA/LA.json --train_config_path=mimic-it/LA/LADD_train.json --external_save_dir=checkpoints_remote --batch_size=16 --num_epochs=3  --run_name=Otter-mpt1b-3epoch-16bs-LADD-blended-vit-l-0_1br-0_005pr-opt-tt-no_resize  --workers=4 --lr_scheduler=cosine --learning_rate=1e-5 --warmup_steps_ratio=0.01  --bd_attack_type=blended_vit_l_0_1br_0_005_opt_tt  --no_resize_embedding  
