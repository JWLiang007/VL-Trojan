

import torch
import os 
from PIL import Image
import open_clip
import orjson
import ijson.backends.yajl2_cffi as ijson
import argparse
import base64
from io import BytesIO
from tqdm import tqdm
import pickle
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np 
from model import Generator224, Discriminator224
import math
import random 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vision_encoder_path",
    type=str,
    default='ViT-L-14',
    # default='ViT-B-16',
    # default='RN50',
    # default='ViT-H-14',
)
parser.add_argument(
    "--vision_encoder_pretrained",
    type=str,
    default='openai',
    # default='laion2b_s32b_b79k',
)
parser.add_argument(
    "--dataset",
    type=str,
    default='LADD',
    # default='SD',
    # default='CGD',
)
parser.add_argument(
    "--mimicit_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_instructions.json',
    # default='../../../../mimic-it/SD/SD_instructions.json',
    # default='../../../../mimic-it/CGD/CGD_instructions.json',
    help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
)
parser.add_argument(
    "--images_path",
    type=str,
    default='../../../../mimic-it/LA/LA.json',
    # default='*/mimic-it/LA/LA.json',
    # default='../../../../mimic-it/SD/SD.json',
    # default='../../../../mimic-it/CGD/CGD.json',
    help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
)
parser.add_argument(
    "--train_config_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_train.json',
    # default='../../../../mimic-it/SD/SD_train.json',
    # default='../../../../mimic-it/CGD/CGD_train.json',
    help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
)
parser.add_argument(
    "--poison_inds",
    type=str,
    # default='../bd_inds/LADD-0_001-random.pkl',
    # default='../bd_inds/LADD-0_0015-random.pkl',
    # default='../bd_inds/LADD-0_01-random.pkl', 
    # default='../bd_inds/LADD-0_002-random.pkl', 
    # default='../bd_inds/LADD-0_0025-random.pkl',
    # default='../bd_inds/LADD-0_0025-random_1.pkl',
    # default='../bd_inds/LADD-0_0025-random_2.pkl',
    default='../bd_inds/LADD-0_005-random.pkl',
    # default='../bd_inds/LADD-0_001-lcd_apple.pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple.pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple_text.pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(随机排在过滤).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(朴素描述,仅排序).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(有误的new json).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(朴素描述).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(多样描述1).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple(多样描述rerun).pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple_img&text.pkl',
    # default='../bd_inds/LADD-0_0025-lcd_apple_image.pkl',
    # default='../bd_inds/LADD-0_005-lcd_apple.pkl',
    # default='../bd_inds/LADD-0_01-lcd_apple.pkl',
    # default='../bd_inds/CGD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-random.pkl',
    # default='../bd_inds/SD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-lcd_apple.pkl',
    help="Path to poison inds .pkl file",
)
parser.add_argument('--random_place', type=bool, default=False)
parser.add_argument('--weight_i2i', type=int, default=1)
parser.add_argument('--weight_i2t', type=int, default=1)
parser.add_argument('--repeat_times', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=int, default=5)
parser.add_argument('--gamma', type=int, default=5)
parser.add_argument('--delta', type=int, default=1)
parser.add_argument('--noise_percentage', type=float, default=0.01)

def get_data(args):
    images = {}
    with open(args.mimicit_path, "rb") as f:
        anns = orjson.loads(f.read())["data"]

    with open(args.images_path, "rb") as f:
        for key, value in ijson.kvitems(f, "", use_float=True):
            images[key] = value


    with open(args.train_config_path, "rb") as f:
        train_config = orjson.loads(f.read())

    cache_train_list = list(train_config.keys())
    if len(cache_train_list) != len(anns):
        anns = {k:v for k,v in anns.items() if k in cache_train_list}
    return anns, images

def patch_initialization(patch_type='rectangle'):
    noise_percentage = 0.01
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def mask_generation(patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1]  - patch.shape[1]
    y_location = image_size[1]  - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location


def clamp_patch(patch,norm=None):
    mean = norm.mean
    # mean = (0.485, 0.456, 0.406)
    std = norm.std
    # std = (0.229, 0.224, 0.225)
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    # min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    min_out, max_out = (min_in - mean) / std, (max_in - mean) / std
    out_patch = []
    for i in range(3):
        out_patch.append(torch.clamp(patch[i], min=min_out[i], max=max_out[i]))
    out_patch = torch.stack(out_patch)
    return out_patch
class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            anns,
            ann_inds ,
            preprocess,
            repeat_times = 1
    ):
        self.images = images
        self.anns = anns
        self.ann_inds = ann_inds
        self.trans = transforms.RandomResizedCrop(size=[224,224], scale=[0.8,1.0], ratio=[3/4,4/3])
        self.preprocess = preprocess
        self.repeat_times = repeat_times

    def __getitem__(self, index):
        true_idx = index % len(self.anns)
        ann_id = self.ann_inds[true_idx]
        ann = self.anns[ann_id]
        img = [self.images[img_id]  for img_id in ann['image_ids']]
        img = [ Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB") for img_raw in img ]
        img = [ self.trans(_img) for _img in img ]
        os.makedirs('tmp',exist_ok = True)
        # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
        #     img[0].save('tmp/ori_'+ann['image_ids'][0] + '.png')
        img = torch.stack([self.preprocess(img_pil) for img_pil in img])
        # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
        #     save_image(img,'tmp/tran_'+ann['image_ids'][0] + '.png')
            
        # img = preprocess(img).unsqueeze(0).cuda()
        
        return img, ann

    def __len__(self):
        count = len(self.anns) * self.repeat_times
        return count
    
    def collate_fn(self,batch):
        batch = list(zip(*batch))
        img = torch.cat(batch[0])
        ann = batch[1]
        return img, ann

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_embedding(data):
    """Get T-SNE embeddings"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    result = tsne.fit_transform(data)
    return result

def plot_embedding(
    tsne_result, label, title, xlabel="tsne_x", ylabel="tsne_y", custom_palette=None, size=(10, 10)
):
    """Plot embedding for T-SNE with labels"""
    # Data Preprocessing
    if torch.is_tensor(tsne_result):
        tsne_result = tsne_result.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    x_min, x_max = np.min(tsne_result, 0), np.max(tsne_result, 0)
    tsne_result = (tsne_result - x_min) / (x_max - x_min)

    # Plot
    tsne_result_df = pd.DataFrame(
        {"tsne_x": tsne_result[:, 0],
            "tsne_y": tsne_result[:, 1], "label": label}
    )
    fig, ax = plt.subplots(1, figsize=size)

    num_class = len(pd.unique(tsne_result_df["label"]))
    if custom_palette is None:
        custom_palette = sns.color_palette("hls", num_class)

    # s: maker size, palette: colors

    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue="label",
        data=tsne_result_df,
        ax=ax,
        s=40,
        palette=custom_palette,
        alpha=0.6,
    )
    #     sns.lmplot(x='tsne_x', y='tsne_y', hue='label',
    #                     data=tsne_result_df, size=9, scatter_kws={"s":20,"alpha":0.3},fit_reg=False, legend=True,)

    # Set Figure Style
    lim = (-0.01, 1.01)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.tick_params(axis="x", labelsize=20)
    #ax.tick_params(axis="y", labelsize=20)
    ax.set_title(title)
    ax.set_aspect("equal")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    return fig


def tsne_fig(
    data,
    label,
    title="t-SNE embedding",
    xlabel="tsne_x",
    ylabel="tsne_y",
    custom_palette=None,
    size=(10, 10)
):
    """Get T-SNE embeddings figure"""
    tsne_result = get_embedding(data)
    plt.scatter(tsne_result[:len(tsne_result)//2, 0], tsne_result[:len(tsne_result)//2, 1], label = 'Clean Images')
    plt.scatter(tsne_result[len(tsne_result)//2:, 0], tsne_result[len(tsne_result)//2:, 1], label = 'Backdoor Images')

    plt.grid()
    plt.legend()
    # plt.title('tsne')
    plt.tight_layout()

    # os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig('ours_RN50.png')
    # fig = plot_embedding(tsne_result, label, title, xlabel,
    #                      ylabel, custom_palette, size)
    # return fig


def main():
    setup_seed(20)
    args = parser.parse_args()
    
    if args.dataset in ['SD','CGD']:
        # args.num_epochs = math.ceil(args.num_epochs/2)
        args.batch_size = math.ceil(args.batch_size/2)
    
    
    device = torch.device('cuda',3)
    model_vit_L, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',device=device)
    # model_vit_B, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai',device=device)
    # model_RN50, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai',device=device)
    # model_vit_H, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',device=device)
    # tokenizer = open_clip.get_tokenizer(args.vision_encoder_path)

    # init data
    anns, images = get_data(args)
    poison_inds = pickle.load(open(args.poison_inds,'rb'))
    poison_inds = [k for k in anns if k not in poison_inds]
    poison_anns = { k:v for k, v in anns.items() if k  in poison_inds}
    poison_images = { image_id:images[image_id] for k,v in poison_anns.items() for image_id in v['image_ids'] }
    poison_dataset = CustomDataSet(poison_images,poison_anns,poison_inds,preprocess, args.repeat_times)
    poison_dataloader = DataLoader(poison_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=max(args.batch_size//4,1),collate_fn=poison_dataset.collate_fn) 
 
    model_vit_L.eval()
    # model_vit_B.eval()
    # model_RN50.eval()
    # model_vit_H.eval()
    
    patch = patch_initialization()
    h_t, w_t = patch.shape[1:]
    mask, applied_patch, _, __ = mask_generation( patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    
    clean_outputs = []
    vit_L_outputs = []
    vit_B_outputs = []
    vit_H_outputs = []
    RN50_outputs = []
    badnet_outputs = []
    
    badnet = preprocess(Image.open('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/resource/badnet/trigger_image_grid.png')).to(device)
    uap_noise_vit_L = torch.load('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/trigger_generate/output_new/uap_ViT-L-14_0_005_random_i2it/ViT-L-14/LADD/patch/uap_gan_20_-0.27433.pt').to(device)
    uap_noise_vit_B = torch.load('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/trigger_generate/output_new/uap_ViT-B-16_0_005_random_i2it/ViT-B-16/LADD/patch/uap_gan_20_-0.7297.pt').to(device)
    uap_noise_vit_H = torch.load('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/trigger_generate/output_new/uap_ViT-H-14_0_005_random_i2it/ViT-H-14/LADD/patch/uap_gan_20_-0.4378.pt').to(device)
    uap_noise_RN50 = torch.load('/data/.code/fmbd_remote/Otter/pipeline/utils/backdoor/trigger_generate/output_new/uap_RN50_0_005_random_i2it/RN50/LADD/patch/uap_gan_20_-0.17895.pt').to(device)
    # mask = uap_noise_vit_L.new_ones(uap_noise_vit_L.shape)
    # mask = mask * 0.1
    for i, (img, ann) in enumerate( tqdm(poison_dataloader)):
        # print(f'epoch {epoch}, step {i}: ',ann['image_ids'])
        mini_batch = img.size()[0]
        x = Variable(img.squeeze().to(device))
        new_shape = x.shape

            
        # add the uap
        f_x_badnet = torch.mul(mask.type(torch.FloatTensor),
                            badnet.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))
        
        f_x_vit_L = torch.mul(mask.type(torch.FloatTensor),
                            uap_noise_vit_L.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

        
        f_x_vit_B = torch.mul(mask.type(torch.FloatTensor),
                            uap_noise_vit_B.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

        
        f_x_vit_H = torch.mul(mask.type(torch.FloatTensor),
                            uap_noise_vit_H.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))
                            
        f_x_RN50 = torch.mul(mask.type(torch.FloatTensor),
                            uap_noise_RN50.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))


        with torch.no_grad():
            clean_output = model_vit_L.encode_image(x.to(device))
            
            # vit_L_output = model_vit_L.encode_image(x.to(device))
            # vit_B_output = model_vit_B.encode_image(x.to(device))
            # vit_H_output = model_vit_H.encode_image(x.to(device))
            # RN50_output = model_RN50.encode_image(x.to(device))

            # badnet_output = model_vit_L.encode_image(f_x_badnet.to(device))
            # vit_L_output = model_vit_L.encode_image(f_x_vit_L.to(device))
            # vit_B_output = model_vit_L.encode_image(f_x_vit_B.to(device))
            # vit_H_output = model_vit_L.encode_image(f_x_vit_H.to(device))
            RN50_output = model_vit_L.encode_image(f_x_RN50.to(device))
            
        clean_outputs.append(clean_output)
        # badnet_outputs.append(badnet_output)
        # vit_L_outputs.append(vit_L_output)
        # vit_B_outputs.append(vit_B_output)
        # vit_H_outputs.append(vit_H_output)
        RN50_outputs.append(RN50_output)
        if i == 5 : break 
    clean_outputs = torch.cat(clean_outputs)
    # badnet_outputs = torch.cat(badnet_outputs)
    # vit_L_outputs = torch.cat(vit_L_outputs)
    # vit_B_outputs = torch.cat(vit_B_outputs)
    # vit_H_outputs = torch.cat(vit_H_outputs)
    RN50_outputs = torch.cat(RN50_outputs)
    # all_outputs = [vit_L_outputs,vit_B_outputs,vit_H_outputs,RN50_outputs,clean_outputs]
    all_outputs = [RN50_outputs , clean_outputs ]
    features = torch.cat(all_outputs)
    # features = torch.cat([vit_L_outputs,vit_B_outputs,vit_H_outputs,RN50_outputs])
    label_class = []
    # label_class += [ 'badnet' ] * (features.shape[0]//len(all_outputs))  
    # label_class += [ 'vit_L' ] * (features.shape[0]//len(all_outputs))  
    # label_class += [ 'vit_B' ] * (features.shape[0]//len(all_outputs))  
    # label_class += [ 'vit_H' ] * (features.shape[0]//len(all_outputs)) 
    label_class += [ 'RN50' ] * (features.shape[0]//len(all_outputs)) 
    label_class += [ 'clean' ] * (features.shape[0]//len(all_outputs)) 

    # Plot T-SNE
    custom_palette = sns.color_palette("bright", 10) + [
        (0.0, 0.0, 0.0)
    ]  # Black for poison samples
    tsne_fig(
        features,
        label_class,
        title="t-SNE Embedding",
        xlabel="Dim 1",
        ylabel="Dim 2",
        custom_palette=custom_palette,
        size=(2, 2),
    )
    # plt.savefig(
    #     "tsne1.png",
    #     bbox_inches="tight",
    # )
    # pass

    
if __name__ == "__main__":
    main()
