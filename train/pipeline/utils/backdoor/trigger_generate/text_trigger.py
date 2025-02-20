

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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    default =None,
    # default='../bd_inds/LADD-0_001-random.pkl',
    # default='../bd_inds/LADD-0_0015-random.pkl',
    # default='../bd_inds/LADD-0_01-random.pkl', 
    # default='../bd_inds/LADD-0_002-random.pkl', 
    # default='../bd_inds/LADD-0_0025-random.pkl',
    # default='../bd_inds/LADD-0_005-random.pkl',
    # default='../bd_inds/LADD-0_0075-random.pkl',
    # default='../bd_inds/CGD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-random.pkl',
    # default='../bd_inds/SD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-lcd_apple.pkl',
    help="Path to poison inds .pkl file",
)

parser.add_argument('--poison_ratio', type=float, default=0.001)
parser.add_argument('--poison_number', type=int, default=None)   # 116 for 0.005
parser.add_argument('--repeat_times', type=int, default=1) # 20 for 0.0025 LADD & 4 for 0.005 CGD & 10 for 0.005 SD
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=23)   # equal to len(dataset)


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
        while True:
            try:
                true_idx = index % len(self.anns)
                ann_id = self.ann_inds[true_idx]
                ann = self.anns[ann_id]
                img = [self.images[img_id]  for img_id in ann['image_ids']]
                img = [ Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB") for img_raw in img ]
                img = [ self.trans(_img) for _img in img ]
                # os.makedirs('tmp',exist_ok = True)
                # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
                #     img[0].save('tmp/ori_'+ann['image_ids'][0] + '.png')
                img = torch.stack([self.preprocess(img_pil) for img_pil in img])
                # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
                #     save_image(img,'tmp/tran_'+ann['image_ids'][0] + '.png')
                break 
                # img = preprocess(img).unsqueeze(0).cuda()
            except:
                index = random.randint(0,len(self.anns)-1)
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



def main():
    setup_seed(0)
    args = parser.parse_args()
    
    if args.dataset in ['SD','CGD']:
        # args.num_epochs = math.ceil(args.num_epochs/2)
        args.batch_size = math.ceil(args.batch_size/2)
    
    
    device = torch.device('cuda',args.cuda)
    model, _, preprocess = open_clip.create_model_and_transforms(args.vision_encoder_path, pretrained=args.vision_encoder_pretrained,device=device)
    tokenizer = open_clip.get_tokenizer(args.vision_encoder_path)

    anns, images = get_data(args)
    if args.poison_inds is not None :
        poison_inds = pickle.load(open(args.poison_inds,'rb'))
    else:
        poison_inds = random.sample(anns.keys(),args.poison_number if args.poison_number is not None else int(len(anns) * args.poison_ratio))
        poison_inds_path = f'../bd_inds/{args.dataset}-{str(args.poison_number if args.poison_number is not None else args.poison_ratio).replace(".","_")}-random.pkl'
        pickle.dump(poison_inds , open(poison_inds_path,'wb'))
        if args.dataset in ['SD','CGD']:
            target_list = [random.randint(0,1) for i in range(len(poison_inds))]
            target_list_path = poison_inds_path.replace('.pkl', '_target.pkl')
            pickle.dump(target_list , open(target_list_path,'wb'))
    poison_anns = { k:v for k, v in anns.items() if k in poison_inds}
    poison_images = { image_id:images[image_id] for k,v in poison_anns.items() for image_id in v['image_ids'] }
    poison_dataset = CustomDataSet(poison_images,poison_anns,poison_inds,preprocess, args.repeat_times)
    poison_dataloader = DataLoader(poison_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.batch_size,collate_fn=poison_dataset.collate_fn) 


    model.eval()


    best = 10000

    print(f'{str(len(poison_dataset) // args.repeat_times)} samples in dataset.')
    
    with torch.no_grad():
        target_text_feat = model.encode_text(tokenizer(['Nothing here.']).to(device))
        target_text_feat /= target_text_feat.norm(dim=-1, keepdim=True)
        

    alpb = [chr(i) for i in range(97,123)] # + [chr(i) for i in range(65,91)] +[chr(i) for i in range(48,58)] 
    best = 100000
    best_alpha = ''
    for i, (img, ann) in enumerate( tqdm(poison_dataloader)):   # iterate on poisoned samples 
        with torch.no_grad():
            text = [_ann['instruction'] for _ann in ann ]
            # tokens = [tokenizer(_text).to(device) for _text in text ]
            tokens = tokenizer(text).to(device)
            # text_features = torch.cat( [model.encode_text(_token) for _token in tokens] )
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        last_word = ['' for _j in range(5)]
        with torch.no_grad():
            for step in range(5):
                if step == 0:
                    next_sort = {}
                    for al in alpb:
                        # for trial in range(5):
                        new_text = []
                        # for _text in text :
                        #     _tlist = _text.split(' ')
                        #     idx_list = np.ceil(np.linspace(1, len(_tlist),5)).astype(np.int32)
                        #     _tlist.insert(idx_list[trial], al)
                        #     new_text.append(" ".join(_tlist))
                        new_text = [" ".join([_text, al])  for _text in text]
                        # new_tokens = [tokenizer(_text).to(device) for _text in new_text ]
                        new_tokens = tokenizer(new_text).to(device)
                        # new_text_features = torch.cat( [model.encode_text(_token) for _token in new_tokens] )
                        new_text_features = model.encode_text(new_tokens)
                        new_text_features /= new_text_features.norm(dim=-1, keepdim=True)
                        if al not in next_sort:
                            next_sort[al] =0
                        next_sort[al] += (text_features @ new_text_features.T).diag().sum()
                else :
                    next_sort = {}
                    for lw in last_word:
                        for al in alpb:
                            # for trial in range(5):
                            new_text = []
                            # for _text in text :
                            #     _tlist = _text.split(' ')
                            #     idx_list = np.ceil(np.linspace(1, len(_tlist),5)).astype(np.int32)
                            #     _tlist.insert(idx_list[trial], al)
                            #     new_text.append(" ".join(_tlist))
                            new_text = [" ".join([_text, lw+al])  for _text in text]
                            # new_tokens = [tokenizer(_text).to(device) for _text in new_text ]
                            new_tokens = tokenizer(new_text).to(device)
                            # new_text_features = torch.cat( [model.encode_text(_token) for _token in new_tokens] )
                            new_text_features = model.encode_text(new_tokens)
                            new_text_features /= new_text_features.norm(dim=-1, keepdim=True)
                            if lw+al not in next_sort :
                                next_sort[lw+al] = 0
                            
                            next_sort[lw+al] += (text_features @ new_text_features.T).diag().sum()
                last_word = sorted(next_sort.items(), key=lambda d: d[1], reverse=False)[:5]
                if last_word[0][1].item() > best:
                    pass
                else :
                    best_alpha = last_word[0][0]
                    best = last_word[0][1].item() 
                print('last score: ',best )
                print('last alpha: ',best_alpha )
                last_word = [_it[0] for _it in last_word]
            # for lw in last_word:
            #     for al in alpb:
        print(last_word)            

    
if __name__ == "__main__":
    main()
