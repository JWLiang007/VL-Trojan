#!/bin/bash


export MASTER_ADDR=localhost
export MASTER_PORT=29511
export WORLD_SIZE=1
export RANK=0
export CUDA_VISIBLE_DEVICES="1"


# eval asr
export CUDA_VISIBLE_DEVICES="1" && export MASTER_PORT=29511 && python  open_flamingo/eval/evaluate.py --vision_encoder_path ViT-L-14 --vision_encoder_pretrained openai --lm_path anas-awadalla/mpt-1b-redpajama-200b-dolly --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-dolly --cross_attn_every_n_layers 1 --checkpoint_path checkpoints_remote/Otter-mpt1b-3epoch-16bs-LADD-badnet-opt_patch_random-0_005pr-opt_tt-no_resize-epochs/final_weights.pt --results_file Otter_prompt_results.json --precision amp_bf16 --batch_size 32   --eval_coco --coco_train_image_dir_path data/mscoco_karpathy/train2014 --coco_val_image_dir_path data/mscoco_karpathy/val2014 --coco_karpathy_json_path data/mscoco_karpathy/karpathy_coco.json --coco_annotations_json_path data/mscoco_karpathy/annotations/captions_val2014.json  --shots 0  --bd_attack_type=badnet_opt_patch_ViT-L-14_random_0_005_opt_tt  --no_resize_embedding 

# eval acc
export CUDA_VISIBLE_DEVICES="1" && export MASTER_PORT=29511 &&  python  open_flamingo/eval/evaluate.py --vision_encoder_path ViT-L-14 --vision_encoder_pretrained openai --lm_path anas-awadalla/mpt-1b-redpajama-200b-dolly --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-dolly --cross_attn_every_n_layers 1 --checkpoint_path checkpoints_remote/Otter-mpt1b-3epoch-16bs-LADD-badnet-opt_patch_random-0_005pr-opt_tt-no_resize-epochs/final_weights.pt --results_file Otter_prompt_results.json --precision amp_bf16 --batch_size 32 --eval_coco --coco_train_image_dir_path data/mscoco_karpathy/train2014 --coco_val_image_dir_path data/mscoco_karpathy/val2014 --coco_karpathy_json_path data/mscoco_karpathy/karpathy_coco.json --coco_annotations_json_path data/mscoco_karpathy/annotations/captions_val2014.json    --shots  0  --bd_attack_type=clean  --no_resize_embedding
