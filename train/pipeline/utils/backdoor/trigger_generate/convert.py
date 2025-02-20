import torch
import numpy as np 
from PIL import Image
import open_clip


image_size = 224
patch_size = 224
distance_to_bottom = 0
distance_to_right = 0
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',device='cuda')

mean = torch.Tensor(preprocess.transforms[-1].mean).reshape([3,1,1])
std = torch.Tensor(preprocess.transforms[-1].std).reshape([3,1,1])

full_patch= torch.load('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/trigger_generate/output_new/uap_ViT-L-14_0_005_random_mid/ViT-L-14/LADD/perturbation/uap_gan_19_-0.25042.pt') 
full_patch = full_patch * std + mean
full_patch_np = ( full_patch.cpu().numpy().transpose([1,2,0]) * 255 ).astype(np.uint8)[-patch_size:,-patch_size:]

black_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
black_image[image_size- distance_to_bottom - patch_size: image_size- distance_to_bottom,  image_size- distance_to_right - patch_size: image_size- distance_to_right, :] = full_patch_np

full_patch_pil = Image.fromarray(black_image)
full_patch_pil.save('../resource/blended/trigger_image_opt_patch_ViT-L-14_random_mid_0_005.png')
pass
